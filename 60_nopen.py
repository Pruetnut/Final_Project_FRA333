import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- 1. CONFIGURATION ---
IMAGE_PATH = "image/Bird.jpg" 

OUTPUT_WAYPOINTS_CSV = "Waypoints_with_z.csv"

# --- Robot Workspace Settings ---
CENTER_X = 0.5              #m
CENTER_Y = 0.0              #m
DRAWING_WIDTH_M = 0.6       #m

Z_DRAW = 0.00               #m
Z_SAFE = 0.05               #m

#Image size & 
IMG_PROCESS_WIDTH = 500     #px
MIN_PATH_PX = 10            #define new path
JUMP_THRESHOLD_M = 0.05     # กระโดดเกิน 1.0 pixel ในแกน X และ Y ถือว่าขึ้นเส้นให
SKIP_PIXEL_STEP = 1         #px reduce pixels

# --- 2. HELPER FUNCTIONS ---
def process_image(path, target_width):
    img = cv2.imread(path, 0)
    if img is None:
        print(f"Warning: Image {path} not found. Using dummy square.")
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, 3)
    
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    
    #reduced noise
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    
    edges_bool = edges > 0
    skeleton_lee = skeletonize(edges_bool, method='lee')
    skeleton_uint8 = (skeleton_lee * 255).astype(np.uint8)
    
    return skeleton_uint8, new_h, target_width

# --- 3. MAIN PIPELINE ---
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Detected {len(contours)} contours.")

# # review the contour
# rows = []
# for idx, cnt in enumerate(contours):
#     for pt in cnt:
#         x, y = pt[0]
#         rows.append([idx, x, y])
# df_contour = pd.DataFrame(rows, columns=["contour_id", "x", "y"])
# df_contour.to_csv("contours_edge2.csv", index=False)
# print(f"Saved {len(df_contour)}")

def sort_contours_nearest(contours, start_pos):
    """Sorts contours using the Nearest Neighbor algorithm."""
    # (Implementation detail provided in the previous turn)
    # Using a simplified version for demonstration here:
    if not contours: return []
    
    sorted_cnts = []
    cnt_data = []
    for cnt in contours:
        if len(cnt) < 2: continue
        pts = cnt.reshape(-1, 2)
        cnt_data.append({'pts': pts, 'start': pts[0], 'end': pts[-1], 'visited': False})
    
    current_pos = np.array(start_pos)
    
    while True:
        nearest_idx = -1
        min_dist = float('inf')
        found_any = False
        for i, item in enumerate(cnt_data):
            if not item['visited']:
                found_any = True
                dist = np.linalg.norm(item['start'] - current_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
        
        if not found_any: break
        cnt_data[nearest_idx]['visited'] = True
        sorted_cnts.append(cnt_data[nearest_idx]['pts'])
        current_pos = cnt_data[nearest_idx]['end']
    return sorted_cnts
# --- END HELPER FUNCTIONS ---


# --- 3. MAIN PIPELINE ---
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
# ใช้ RETR_EXTERNAL เพื่อให้ได้เส้นสะอาดกว่า RETR_LIST ในบางกรณี
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
print(f"Detected {len(contours)} raw contours.")

# 1. Setup Mapping
scale_factor = DRAWING_WIDTH_M / img_w
pixel_center_x = img_w / 2
pixel_center_y = img_h / 2

def to_robot(px, py):
    # Mapping formula (X forward, Y side)
    shifted_x = px - pixel_center_x
    shifted_y = py - pixel_center_y
    rx = CENTER_X + (shifted_y * scale_factor) 
    ry = CENTER_Y + (shifted_x * scale_factor) 
    return rx, ry

# 2. Sort Contours (Start from top-center pixel)
sorted_contours = sort_contours_nearest(contours, start_pos=(pixel_center_x, 0))

# 3. Generate Waypoints (The Core Sequence Logic)
key_waypoints = []
last_x, last_y = CENTER_X, CENTER_Y # Start at general Home
last_z = Z_SAFE # Assume robot starts lifted

# 3.1 Loop through sorted contours
for i, raw_points in enumerate(sorted_contours):
    
    # Downsample points
    points = raw_points[::SKIP_PIXEL_STEP].reshape(-1, 2)
    if len(points) < 5: continue # Filter very short lines

    # Convert start point to robot coordinates
    sx, sy = to_robot(points[0, 0], points[0, 1])
    
    # --- Check for Lift/Drop Logic (The Z-Hop Decision) ---
    dist_to_start = np.linalg.norm(np.array([sx, sy]) - np.array([last_x, last_y]))
    
    # If this is not the first contour AND the move is long enough to warrant a lift
    if i > 0 and dist_to_start > JUMP_THRESHOLD_M:
        
        # A. Lift (ยกขึ้นที่ตำแหน่งเดิม)
        key_waypoints.append({'x': last_x, 'y': last_y, 'z': Z_SAFE, 'type': 0, 'cmd': 'LIFT'})
        
        # B. Travel (บินข้ามไปเหนือจุดเริ่มใหม่)
        key_waypoints.append({'x': sx, 'y': sy, 'z': Z_SAFE, 'type': 0, 'cmd': 'TRAVEL'})
        
        # C. Lower (วางปากกาลงจุดเริ่ม)
        key_waypoints.append({'x': sx, 'y': sy, 'z': Z_DRAW, 'type': 0, 'cmd': 'LOWER'})
        
        last_z = Z_DRAW
        
    elif i == 0:
        # First contour: just move down from Z_SAFE (if not already there)
        if last_z != Z_DRAW:
            key_waypoints.append({'x': sx, 'y': sy, 'z': Z_SAFE, 'type': 0, 'cmd': 'HOVER_START'})
            key_waypoints.append({'x': sx, 'y': sy, 'z': Z_DRAW, 'type': 0, 'cmd': 'LOWER'})
            last_z = Z_DRAW
        
    # 3.2 Draw Loop (Type 1)
    for p in points: 
        rx, ry = to_robot(p[0], p[1])
        key_waypoints.append({'x': rx, 'y': ry, 'z': Z_DRAW, 'type': 1, 'cmd': 'DRAW'})
    
    # Update last position to the end of the stroke
    last_x, last_y = to_robot(points[-1, 0], points[-1, 1])
    last_z = Z_DRAW # Pen remains down

# 3.3 Final Lift and Save
key_waypoints.append({'x': last_x, 'y': last_y, 'z': Z_SAFE, 'type': 0, 'cmd': 'LIFT_END'})
key_waypoints.append({'x': CENTER_X, 'y': CENTER_Y, 'z': Z_SAFE, 'type': 0, 'cmd': 'HOME_END'})


# --- 4. SAVE & PLOT ---
df = pd.DataFrame(key_waypoints)
df.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print(f"✅ Saved {len(df)} waypoints to {OUTPUT_WAYPOINTS_CSV}")