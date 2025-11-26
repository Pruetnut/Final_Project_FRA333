# ============================================================
# 0. PARAMETERS
# ============================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Robot timing (ถ้าจะใช้ trajectory later)
UR5_DT = 0.008
V_MAX = 0.15
A_MAX = 0.3

# Drawing scale & offset (meters)
CANVAS_WIDTH_M = 0.50
CANVAS_HEIGHT_M = 0.50

OFFSET_X = 0.40      # ระยะห่างจากฐานหุ่นยนต์
OFFSET_Y = -0.25     # ขยับภาพไปทางซ้ายเพื่อเริ่มที่มุมกระดาษ

# Pen heights (robot usage)
Z_UP = 0.05
Z_DOWN = 0.00

# ============================================================
# 1. IMAGE → EDGE DETECTION
# ============================================================

def process_image_to_edges(image_path):
    """load → resize → blur → detect edge"""

    img = cv2.imread(image_path, 0)
    h, w = img.shape[:2]

    # Resize ให้ความกว้างเหลือ 600px
    new_w = 600
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_w, new_h))

    # smoothing ลด noise
    g_blur = cv2.GaussianBlur(img, (5, 5), 0)
    bilateral = cv2.bilateralFilter(g_blur, 15, 75, 75)

    # edge detection
    edges = cv2.Canny(bilateral, 50, 150)

    return edges, new_h, new_w

# ============================================================
# 2. EDGE → CONTOUR → SIMPLIFY
# ============================================================

def extract_contours_as_paths(edge_image, min_area=100):
    """Turn edge pixels → contour paths (each path = list of points)"""

    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_paths = []

    for cnt in contours:

        if cv2.contourArea(cnt) > min_area or cv2.arcLength(cnt, False) > 50:

            # simplify contour
            epsilon = 0.001 * cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, epsilon, False)

            valid_paths.append(approx.reshape(-1, 2))

    return valid_paths

# ============================================================
# 3. PIXEL → METER TRANSFORMATION
# ============================================================

def transform_and_sort_paths(pixel_paths, scale_x, scale_y, img_height, offset_x, offset_y):
    robot_paths = []

    # 3.1 Convert pixel → meter → add offset → flip Y
    for path in pixel_paths:
        new_path = []
        for (u, v) in path:
            x_m = u * scale_x
            y_m = v * scale_y

            # flip Y (image origin = top-left)
            y_m_flipped = (img_height * scale_y) - y_m

            # add robot offset
            final_x = x_m + offset_x
            final_y = y_m_flipped + offset_y

            new_path.append([final_x, final_y])
        robot_paths.append(new_path)

    # 3.2 Path sorting using nearest start point (Greedy)
    sorted_paths = []
    unvisited = robot_paths.copy()
    current_pos = (offset_x, offset_y)

    while unvisited:
        nearest_i = -1
        nearest_dist = float('inf')

        for i, path in enumerate(unvisited):
            px, py = path[0]
            d = math.dist(current_pos, (px, py))

            if d < nearest_dist:
                nearest_dist = d
                nearest_i = i

        chosen = unvisited.pop(nearest_i)
        sorted_paths.append(chosen)
        current_pos = chosen[-1]

    return sorted_paths

# ============================================================
# 4. VISUALIZATION
# ============================================================

def visualize_paths_pixel(paths):
    plt.figure(figsize=(8, 6))
    for p in paths:
        plt.plot(p[:, 0], p[:, 1], ".", markersize=1)
    plt.gca().invert_yaxis()
    plt.title("Contours (Pixel)")
    plt.axis("equal")
    plt.show()

def visualize_paths_robot(paths):
    plt.figure(figsize=(8, 8))
    for i, path in enumerate(paths):
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        plt.plot(x, y, "-", linewidth=1)
    plt.title("Robot Paths (Meter)")
    plt.axis("equal")
    plt.show()

# ============================================================
# RUN FULL PIPELINE
# ============================================================

IMAGE_PATH = "image/FIBO.png"
# IMAGE_PATH = "image/line 2.png"

# step 1
edges, h, w = process_image_to_edges(IMAGE_PATH)
scale_x = CANVAS_WIDTH_M / w
scale_y = CANVAS_HEIGHT_M / h

cv2.imshow("Edges", edges)
cv2.waitKey(1)

# step 2
paths_pixel = extract_contours_as_paths(edges)
visualize_paths_pixel([np.array(p) for p in paths_pixel])

# step 3
paths_robot = transform_and_sort_paths(
    paths_pixel,
    scale_x, scale_y,
    h,
    OFFSET_X, OFFSET_Y
)

# step 4
visualize_paths_robot(paths_robot)

# show sample result
print("Total paths:", len(paths_robot))
print("Example point:", paths_robot[0][0])
