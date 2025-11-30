import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.util import invert
from skimage.morphology import skeletonize
from skimage import data

# --- 1. CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png"
OUTPUT_WAYPOINTS_CSV = "Matlab/Waypoint_form_75.csv"

# --- Robot Workspace ---
CENTER_X = 0.5      # เมตร
CENTER_Y = 0.0      # เมตร
DRAWING_WIDTH_M = 0.5  # เมตร

# Z Levels (Standard Z-Up)
Z_DRAW = 0.00       # วาด (ต่ำ)
Z_SAFE = 0.08       # ยก (สูง)

IMG_PROCESS_WIDTH = 500
SKIP_PIXEL_STEP = 4 

# --- 2. HELPER FUNCTIONS ---

def process_image(path, target_width):
    img = cv2.imread(path, 0)
    if img is None:
        print("Image not found, creating dummy.")
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, 3)
    
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    
    kernel = np.ones((3,3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    binary_img = thick_edges > 0
    skeleton = skeletonize(binary_img)
    
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    return skeleton_uint8, new_h, target_width

def sort_contours_nearest(contours, start_pos=(0,0)):
    """
    เรียงลำดับเส้น: เส้นถัดไปต้องมีจุดเริ่ม ใกล้กับจุดจบของเส้นปัจจุบันมากที่สุด
    """
    if not contours: return []
    
    sorted_cnts = []
    # แปลง contours ให้จัดการง่ายขึ้น (List of start/end points)
    # เก็บข้อมูล: [original_contour, start_pt, end_pt]
    cnt_data = []
    for cnt in contours:
        if len(cnt) < 2: continue
        pts = cnt.reshape(-1, 2)
        cnt_data.append({
            'pts': pts,
            'start': pts[0],
            'end': pts[-1],
            'visited': False
        })
    
    current_pos = np.array(start_pos)
    
    while True:
        nearest_idx = -1
        min_dist = float('inf')
        
        # หาเส้นที่ใกล้ที่สุดที่ยังไม่เคยไป
        found_any = False
        for i, item in enumerate(cnt_data):
            if not item['visited']:
                found_any = True
                # เช็คระยะจาก current_pos ไปยัง start ของเส้นนี้
                dist = np.linalg.norm(item['start'] - current_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
        
        if not found_any: break # ครบทุกเส้นแล้ว
        
        # Mark as visited
        cnt_data[nearest_idx]['visited'] = True
        sorted_cnts.append(cnt_data[nearest_idx]['pts'])
        
        # Update current pos ไปที่จุดจบของเส้นที่เพิ่งเลือก
        current_pos = cnt_data[nearest_idx]['end']
        
    return sorted_cnts

# --- 3. MAIN PIPELINE ---

# A. Image Processing
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Original Contours: {len(contours)}")

# B. Mapping Setup
scale_factor = DRAWING_WIDTH_M / img_w
pixel_center_x = img_w / 2
pixel_center_y = img_h / 2

def to_robot(px, py):
    shifted_x = px - pixel_center_x
    shifted_y = py - pixel_center_y
    rx = CENTER_X + (shifted_y * scale_factor) 
    ry = CENTER_Y + (shifted_x * scale_factor) 
    return rx, ry

# C. Sorting Contours (Optimization)
# แปลง Start Point ของ Robot กลับเป็น Pixel คร่าวๆ เพื่อเริ่มหาจากตรงนั้น
# หรือเริ่มจากมุมภาพ (0,0)
sorted_contours = sort_contours_nearest(contours, start_pos=(0,0))
print(f"Sorted Contours: {len(sorted_contours)}")


# D. Generate Sequence with EXPLICIT COMMANDS
waypoints = []

# Initial Home
waypoints.append({'x': CENTER_X, 'y': CENTER_Y, 'z': Z_SAFE, 'type': 0, 'cmd': 'HOME'})

last_x, last_y = CENTER_X, CENTER_Y

for raw_points in sorted_contours:
    points = raw_points[::SKIP_PIXEL_STEP]
    if not np.array_equal(points[-1], raw_points[-1]):
        points = np.vstack((points, raw_points[-1]))
    if len(points) < 2: continue

    start_px, start_py = points[0]
    sx, sy = to_robot(start_px, start_py)
    
    # 1. LIFT (ยกที่เดิม)
    waypoints.append({'x': last_x, 'y': last_y, 'z': Z_SAFE, 'type': 0, 'cmd': 'LIFT'})
    
    # 2. TRAVEL (ย้ายไปเหนือจุดใหม่)
    waypoints.append({'x': sx, 'y': sy, 'z': Z_SAFE, 'type': 0, 'cmd': 'TRAVEL'})
    
    # 3. LOWER (วางลง)
    waypoints.append({'x': sx, 'y': sy, 'z': Z_DRAW, 'type': 0, 'cmd': 'LOWER'})
    
    # 4. DRAW (วาด)
    for p in points[1:]:
        px, py = p
        rx, ry = to_robot(px, py)
        # Type 1 คือ Draw
        waypoints.append({'x': rx, 'y': ry, 'z': Z_DRAW, 'type': 1, 'cmd': 'DRAW'})
    
    last_px, last_py = points[-1]
    last_x, last_y = to_robot(last_px, last_py)

# Final Lift & Home
waypoints.append({'x': last_x, 'y': last_y, 'z': Z_SAFE, 'type': 0, 'cmd': 'LIFT'})
waypoints.append({'x': -0.3, 'y': 0.2, 'z': 0.7, 'type': 0, 'cmd': 'HOME'})

# Save (เพิ่ม column 'cmd')
df = pd.DataFrame(waypoints)
df.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print("Saved with commands.")

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xs = df['x']
ys = df['y']
zs = df['z']

# วาดเส้นเชื่อมให้เห็นลำดับ (Sequence)
ax.plot(xs, ys, zs, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

colors = ['red' if t == 0 else 'blue' for t in df['type']]
ax.scatter(xs, ys, zs, c=colors, s=5)

ax.set_title("Ordered Waypoints (Sorted + Explicit Transitions)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_box_aspect([1,1,0.5])

plt.show()