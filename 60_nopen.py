import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- 1. CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png" 
OUTPUT_WAYPOINTS_CSV = "Waypoints_For_Ruckig.csv"

# --- Robot Workspace Settings ---
CENTER_X = 0.5      
CENTER_Y = 0.0      
DRAWING_WIDTH_M = 0.6 

# ตั้งค่า Z ให้เท่ากัน (เพื่อไม่ให้ยก)
Z_LEVEL = 0.00      # ระดับเดียวตลอดกาล

IMG_PROCESS_WIDTH = 500
SKIP_PIXEL_STEP = 4 

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
    
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    
    edges_bool = edges > 0
    skeleton_lee = skeletonize(edges_bool, method='lee')
    skeleton_uint8 = (skeleton_lee * 255).astype(np.uint8)
    
    return skeleton_uint8, new_h, target_width

# --- 3. MAIN PIPELINE ---

edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
# cv2.imshow("Skeleton Edges", edges)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Detected {len(contours)} contours.")

scale_factor = DRAWING_WIDTH_M / img_w
pixel_center_x = img_w / 2
pixel_center_y = img_h / 2

key_waypoints = []

# จุด Home
key_waypoints.append({'x': CENTER_X, 'y': CENTER_Y, 'z': Z_LEVEL, 'type': 0})

total_points = 0

for i, cnt in enumerate(contours):
    if len(cnt) < 15: continue
    
    raw_points = cnt.reshape(-1, 2)
    
    # Downsampling
    points = raw_points[::SKIP_PIXEL_STEP]
    if not np.array_equal(points[-1], raw_points[-1]):
        points = np.vstack((points, raw_points[-1]))
        
    total_points += len(points)
    
    def transform_pixel_to_robot(px, py):
        shifted_x = px - pixel_center_x
        shifted_y = py - pixel_center_y
        rx = CENTER_X + (shifted_y * scale_factor) 
        ry = CENTER_Y + (shifted_x * scale_factor) 
        return rx, ry

    # --- Generate Path Sequence (Logic ใหม่: ไม่มียก) ---
    
    # Start Point
    start_px, start_py = points[0]
    sx, sy = transform_pixel_to_robot(start_px, start_py)
    
    # A. Travel to Start (Type 0 = Stop/Travel)
    # วิ่งจากจุดเดิม มาหาจุดเริ่มเส้นใหม่ (ที่ระดับความสูงเดียวกัน)
    key_waypoints.append({'x': sx, 'y': sy, 'z': Z_LEVEL, 'type': 0})
    
    # *ตัด B. Pen Down ทิ้ง* (เพราะ Z เท่าเดิม ไม่ต้องกดซ้ำ)
    
    # C. Draw Loop (Type 1 = Continuous)
    for p in points[1:]:
        px, py = p
        rx, ry = transform_pixel_to_robot(px, py)
        key_waypoints.append({'x': rx, 'y': ry, 'z': Z_LEVEL, 'type': 1})
        
    # *ตัด D. Pen Up ทิ้ง* (เพราะ Z เท่าเดิม ไม่ต้องยกซ้ำ)
    
    # (Optional) ถ้าอยากให้จบเส้นแล้วหยุดนิดนึง ค่อยเพิ่มจุดสุดท้ายเป็น Type 0
    # last_px, last_py = points[-1]
    # lx, ly = transform_pixel_to_robot(last_px, last_py)
    # key_waypoints.append({'x': lx, 'y': ly, 'z': Z_LEVEL, 'type': 0})

# --- 4. SAVE & PLOT ---
df = pd.DataFrame(key_waypoints)
df.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print(f"Saved {len(df)} waypoints to {OUTPUT_WAYPOINTS_CSV}")

# Visualization 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xs = df['x']
ys = df['y']
zs = df['z']

# สีแดง = Travel (ช่วงกระโดดข้ามเส้น), สีน้ำเงิน = Draw
colors = ['red' if t == 0 else 'blue' for t in df['type']]
sizes = [20 if t == 0 else 5 for t in df['type']]

ax.scatter(xs, ys, zs, c=colors, s=sizes)

# วาดเส้นเชื่อมให้เห็นการลาก (Drag Line)
ax.plot(xs, ys, zs, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Robot Origin
ax.scatter(0, 0, 0, c='k', s=100, marker='^', label='Robot Base')

ax.set_title("Robot Path (No Z-Hop / Dragging)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_box_aspect([1,1,0.5])

plt.legend()
plt.show()