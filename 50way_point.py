import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png" 
OUTPUT_WAYPOINTS_CSV = "Waypoints_For_Ruckig.csv" # ชื่อไฟล์สำหรับเอาไปใส่ Ruckig

# Workspace & Robot Settings
CANVAS_WIDTH_M = 0.5        
IMG_PROCESS_WIDTH = 400     
Z_SAFE = 0.05               
Z_DRAW = 0.00               

OFFSET_X = 0.3              
OFFSET_Y = 0.3              

# --- TUNING PARAMETER (สำคัญตรงนี้) ---
# ค่านี้ยิ่งมาก = จุดยิ่งน้อย (รูปเหลี่ยมขึ้น)
# ค่านี้ยิ่งน้อย = จุดยิ่งเยอะ (รูปโค้งเนียนเหมือนเดิม)
# แนะนำ: 0.001 ถึง 0.005
APPROX_EPSILON_FACTOR = 0.005

# --- HELPER FUNCTIONS ---
# (ตัด generate_segment ออก เพราะเราจะให้ Ruckig เป็นคนทำ)

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
    
    # Preprocessing
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur,15,75,75)
    edges = cv2.Canny(bilateral,50,150)
    skeleton_lee = skeletonize(edges, method='lee')
    skeleton_uint8 = (skeleton_lee * 255).astype(np.uint8)
    return skeleton_uint8, new_h, target_width

# --- MAIN PIPELINE ---

# 1. Image Processing
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
cv2.imshow("skeleton", edges)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 2. Prepare Scaling
scale_factor = CANVAS_WIDTH_M / img_w
offset_x = (-CANVAS_WIDTH_M / 2 ) + OFFSET_X
offset_y = ((img_h * scale_factor) / 2 ) + OFFSET_Y

print(f"Detected {len(contours)} contours.")

# 3. Key Waypoints Generation Loop (สร้างลายแทงจุด)
key_waypoints = []
current_pos = np.array([offset_x, offset_y, Z_SAFE]) 

# 3.1 ใส่จุด Home (จุดแรกสุด)
key_waypoints.append({
    'x': current_pos[0], 'y': current_pos[1], 'z': current_pos[2], 'type': 'HOME'
})

total_points_original = 0
total_points_reduced = 0

for i, cnt in enumerate(contours):
    if len(cnt) < 15: continue
    total_points_original += len(cnt)
    
    # 1. คำนวณความยาวเส้นรอบรูป
    epsilon = APPROX_EPSILON_FACTOR * cv2.arcLength(cnt, True)
    # 2. คำนวณเส้นใหม่ที่ลดจุดแล้ว (Approx)
    approx_cnt = cv2.approxPolyDP(cnt, epsilon, closed=False) # closed=False ถ้าเป็นเส้นวาด
    
    total_points_reduced += len(approx_cnt)
    
    points = cnt.reshape(-1, 2)
    
    # คำนวณพิกัดจุดเริ่มต้น (Start Point)
    start_pixel = points[0]
    start_x = (start_pixel[0] * scale_factor) + offset_x
    start_y = offset_y - (start_pixel[1] * scale_factor)
    
    # A. TRAVEL: เคลื่อนไปเหนือจุดเริ่ม (Z_SAFE)
    key_waypoints.append({
        'x': start_x, 'y': start_y, 'z': Z_SAFE, 'type': 0
    })
    
    # B. PEN DOWN: กดปากกาลง (Z_DRAW)
    key_waypoints.append({
        'x': start_x, 'y': start_y, 'z': Z_DRAW, 'type': 0
    })
    
    # C. DRAWING: เก็บจุดทั้งหมดในเส้น (Z_DRAW)
    # Ruckig จะทำหน้าที่วิ่งผ่านจุดเหล่านี้ให้เอง
    for p in points[1:]:
        px = (p[0] * scale_factor) + offset_x
        py = offset_y - (p[1] * scale_factor)
        key_waypoints.append({'x': px, 'y': py, 'z': Z_DRAW, 'type': 1})
        
    # D. PEN UP: ยกปากกาขึ้นเมื่อจบเส้น (Z_SAFE)
    # จุดสุดท้ายของเส้น
    last_pixel = points[-1]
    last_x = (last_pixel[0] * scale_factor) + offset_x
    last_y = offset_y - (last_pixel[1] * scale_factor)
    
    key_waypoints.append({
        'x': last_x, 'y': last_y, 'z': Z_SAFE, 'type': 0
    })

# สรุปผลการลดจุด
print(f"--- Reduction Summary ---")
print(f"Original Points (Raw): {total_points_original}")
print(f"Reduced Points (Approx): {total_points_reduced}")
print(f"Reduction Ratio: {(1 - total_points_reduced/total_points_original)*100:.2f}% removed")

# --- 4. Save CSV ---
df = pd.DataFrame(key_waypoints)
df.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print(f"Generated {len(df)} Key Waypoints.")
print(f"Saved to {OUTPUT_WAYPOINTS_CSV}")
print(df.head(10)) # โชว์ตัวอย่างข้อมูล

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xs = df['x']
ys = df['y']
zs = df['z']

ax.plot(xs, ys, zs, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
colors = ['red' if z > (Z_DRAW + 0.001) else 'blue' for z in zs]
ax.scatter(xs, ys, zs, c=colors, s=10) # เพิ่มขนาดจุด s=10 จะได้เห็นชัดว่าจุดน้อยลง

ax.set_title(f"Optimized Waypoints Map ({len(df)} points)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

mid_x = (xs.max() + xs.min()) * 0.5
mid_y = (ys.max() + ys.min()) * 0.5
max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min()]).max() / 2.0
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(Z_DRAW - 0.01, Z_SAFE + 0.02)

plt.show()


# # --- 5. Visualization 3D (Map Preview) ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# xs = df['x']
# ys = df['y']
# zs = df['z']

# # วาดเส้นเชื่อมต่อจุด เพื่อให้เห็นลำดับการเดินทาง (Sequence)
# ax.plot(xs, ys, zs, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Sequence Path')

# # พลอตจุด (แยกสีตามประเภท)
# # HOME/TRAVEL/PEN_UP = สีแดง (อยู่สูง)
# # DRAW/PEN_DOWN = สีน้ำเงิน (อยู่ต่ำ)
# colors = ['red' if z > (Z_DRAW + 0.001) else 'blue' for z in zs]
# ax.scatter(xs, ys, zs, c=colors, s=5)

# # จุด Home (ใหญ่พิเศษ)
# ax.scatter(xs[0], ys[0], zs[0], c='green', s=100, marker='*', label='Start (Home)')

# ax.set_title(f"Waypoints Map for Ruckig ({len(df)} points)")
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")

# # Scale adjustment
# mid_x = (xs.max() + xs.min()) * 0.5
# mid_y = (ys.max() + ys.min()) * 0.5
# max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min()]).max() / 2.0
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(Z_DRAW - 0.01, Z_SAFE + 0.02)

# plt.legend()
# plt.show()
