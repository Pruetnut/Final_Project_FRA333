import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- 1. CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png" 
OUTPUT_WAYPOINTS_CSV = "Waypoints_For_Ruckig.csv"

# --- Robot Workspace Settings (Standard Z-Up) ---
CENTER_X = 0.5      # ระยะยื่นไปข้างหน้า (เมตร) - ปรับตามจริง
CENTER_Y = 0.0      # ระยะด้านข้าง (เมตร)
DRAWING_WIDTH_M = 0.5  # ความกว้างจริงของรูป (เมตร) - ลองปรับให้พอดีกระดาษ

# ตั้งค่า Z (แกน Z ชี้ขึ้นฟ้า) -> มียกปากกา
Z_DRAW = 0.00       # ระดับหัวปากกาแตะกระดาษ (0 เมตร)
Z_SAFE = 0.05       # ระดับยกปากกา (สูงขึ้นมา 5 ซม.)

IMG_PROCESS_WIDTH = 400
SKIP_PIXEL_STEP = 5  # ลดจุด: เก็บ 1 ข้าม 3

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
    
    # Preprocessing (ตามที่คุณต้องการ)
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    
    # Skeletonize
    edges_bool = edges > 0
    skeleton_lee = skeletonize(edges_bool, method='lee')
    skeleton_uint8 = (skeleton_lee * 255).astype(np.uint8)
    
    return skeleton_uint8, new_h, target_width

# --- 3. MAIN PIPELINE ---

# 1. Image Processing
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)

# แสดงรูป Skeleton เช็คผลลัพธ์
cv2.imshow("Skeleton Result", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# หา Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # ใช้ EXTERNAL เพื่อลดเส้นซ้อนทับ หรือ LIST ตามเดิมก็ได้
print(f"Detected {len(contours)} contours.")

# 2. Mapping Logic
scale_factor = DRAWING_WIDTH_M / img_w
pixel_center_x = img_w / 2
pixel_center_y = img_h / 2

key_waypoints = []

# เพิ่มจุด Home (เริ่มที่ระดับปลอดภัย Z_SAFE)
key_waypoints.append({'x': CENTER_X, 'y': CENTER_Y, 'z': Z_SAFE, 'type': 0})

total_points_original = 0
total_points_reduced = 0

for i, cnt in enumerate(contours):
    if len(cnt) < 15: continue
    
    raw_points = cnt.reshape(-1, 2)
    total_points_original += len(raw_points)
    
    # --- 3. Downsampling (Logic ลดจุดแบบข้าม Pixel) ---
    points = raw_points[::SKIP_PIXEL_STEP]
    
    # เก็บจุดสุดท้ายไว้เสมอ
    if not np.array_equal(points[-1], raw_points[-1]):
        points = np.vstack((points, raw_points[-1]))
        
    total_points_reduced += len(points)

    # --- 4. Coordinate Transformation (Logic หมุนแกนแบบล่าสุด) ---
    def transform_pixel_to_robot(px, py):
        # Shift to center
        shifted_x = px - pixel_center_x
        shifted_y = py - pixel_center_y
        
        # Mapping: Image X->Robot Y, Image Y->Robot X
        # (หมุนภาพให้วางนอนบนโต๊ะตามทิศหุ่นยนต์)
        rx = CENTER_X + (shifted_y * scale_factor) 
        ry = CENTER_Y + (shifted_x * scale_factor) 
        return rx, ry

    # --- Generate Path Sequence (มียกปากกา) ---
    
    # Start Point
    start_px, start_py = points[0]
    sx, sy = transform_pixel_to_robot(start_px, start_py)
    
    # A. Travel (Z_SAFE) - วิ่งมาเหนือจุดเริ่ม
    key_waypoints.append({'x': sx, 'y': sy, 'z': Z_SAFE, 'type': 0})
    
    # B. Pen Down (Z_DRAW) - กดปากกาลง
    key_waypoints.append({'x': sx, 'y': sy, 'z': Z_DRAW, 'type': 0})
    
    # C. Draw Loop (Z_DRAW) - ลากเส้น
    for p in points[1:]:
        px, py = p
        rx, ry = transform_pixel_to_robot(px, py)
        # Type 1 = Draw (Continuous)
        key_waypoints.append({'x': rx, 'y': ry, 'z': Z_DRAW, 'type': 1})
        
    # D. Pen Up (Z_SAFE) - ยกปากกาขึ้นเมื่อจบเส้น
    last_px, last_py = points[-1]
    lx, ly = transform_pixel_to_robot(last_px, last_py)
    key_waypoints.append({'x': lx, 'y': ly, 'z': Z_SAFE, 'type': 0})

# สรุปผลการลดจุด
print(f"--- Reduction Summary (Step={SKIP_PIXEL_STEP}) ---")
print(f"Original Points: {total_points_original}")
print(f"Reduced Points : {total_points_reduced}")
print(f"Reduction Ratio: {(1 - total_points_reduced/total_points_original)*100:.2f}% removed")

# --- 5. SAVE & PLOT ---
df = pd.DataFrame(key_waypoints)
df.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print(f"Saved {len(df)} waypoints to {OUTPUT_WAYPOINTS_CSV}")

# Visualization 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xs = df['x']
ys = df['y']
zs = df['z']

# สีแดง = Travel/Stop (อยู่สูง), สีน้ำเงิน = Draw (อยู่ต่ำ)
colors = ['red' if t == 0 else 'blue' for t in df['type']]
sizes = [20 if t == 0 else 5 for t in df['type']]

ax.scatter(xs, ys, zs, c=colors, s=sizes)

# Plot Table Reference at Z_DRAW
xx, yy = np.meshgrid(np.linspace(CENTER_X-0.2, CENTER_X+0.2, 2), np.linspace(CENTER_Y-0.2, CENTER_Y+0.2, 2))
zz = np.zeros_like(xx) + Z_DRAW 
ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

# Robot Origin
ax.scatter(0, 0, 0, c='k', s=100, marker='^', label='Robot Base (0,0,0)')

ax.set_title("Robot Path (Skeleton + Downsample + Z-Hop + Correct Axis)")
ax.set_xlabel("X (Forward)"); ax.set_ylabel("Y (Side)"); ax.set_zlabel("Z (Height)")
ax.set_box_aspect([1,1,0.5])

plt.legend()
plt.show()