import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png" # เปลี่ยน path รูปของคุณ
OUTPUT_CSV = "UR5_Trajectory_TimeSampled.csv"

# Workspace & Robot Settings
CANVAS_WIDTH_M = 0.2        # ความกว้างจริงของรูปที่จะวาด (เมตร) e.g., 20cm
IMG_PROCESS_WIDTH = 500     # ขนาดภาพสำหรับ Process
Z_SAFE = 0.05               # ระดับยกปากกา (เมตร)
Z_DRAW = 0.00               # ระดับวาด (เมตร)

# Trajectory Settings
SAMPLING_TIME = 0.008       # (seconds) 8ms สำหรับ UR5 default control loop
SPEED_TRAVEL = 0.15         # (m/s) ความเร็วตอนยกเคลื่อนที่
SPEED_DRAW = 0.05           # (m/s) ความเร็วตอนวาดจริง (ช้าๆ จะสวย)

OFFSET_X = 0.2              #(m) เลื่อนจากจุด 0 เท่าไร
OFFSET_Y = 0.2              #(m) เลื่อนจากจุด 0 เท่าไร

# --- HELPER FUNCTIONS ---

def generate_segment(start_pos, end_pos, speed, dt):
    """
    สร้างจุดย่อยๆ ระหว่าง 2 จุด ตาม Sampling Time และความเร็ว
    """
    dist = np.linalg.norm(end_pos - start_pos)
    
    # กรณีจุดซ้ำกัน หรือระยะทางเป็น 0
    if dist < 1e-6:
        return np.array([start_pos])

    # คำนวณเวลาที่ต้องใช้ในการเคลื่อนที่นี้
    duration = dist / speed
    
    # คำนวณจำนวนจุด (Steps) ตาม Sampling Time
    num_steps = int(np.ceil(duration / dt))
    
    if num_steps < 1:
        num_steps = 1
        
    # สร้างจุด Linear Interpolation (x, y, z)
    # linspace จะแบ่งช่วงเท่าๆ กัน
    xs = np.linspace(start_pos[0], end_pos[0], num_steps)
    ys = np.linspace(start_pos[1], end_pos[1], num_steps)
    zs = np.linspace(start_pos[2], end_pos[2], num_steps)
    
    # รวมเป็น Matrix (N, 3)
    segment_points = np.vstack((xs, ys, zs)).T
    return segment_points

def process_image(path, target_width):
    img = cv2.imread(path, 0)
    if img is None:
        # สร้างรูป Dummy ถ้าหาไฟล์ไม่เจอ
        print(f"Warning: Image {path} not found. Using dummy square.")
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, 3)
    
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    
    # Preprocessing
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges, new_h, target_width

# --- MAIN PIPELINE ---

# 1. Image Processing
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # ใช้ EXTERNAL เพื่อเอาแค่เส้นนอก หรือ LIST เอาหมด

# 2. Prepare Scaling
scale_factor = CANVAS_WIDTH_M / img_w
# ตั้งจุด Origin (0,0) ของหุ่นยนต์ให้เริ่มที่มุมซ้ายล่างของรูป หรือตรงกลางตามต้องการ
offset_x = (-CANVAS_WIDTH_M / 2 ) + OFFSET_X
offset_y = ((img_h * scale_factor) / 2 ) + OFFSET_Y

print(f"Detected {len(contours)} contours.")

# 3. Trajectory Generation Loop
full_trajectory_points = []
current_pos = np.array([offset_x, offset_y, Z_SAFE]) # เริ่มที่ Home

full_trajectory_points.append(current_pos.reshape(1, 3))

for i, cnt in enumerate(contours):
    # กรองเส้นที่สั้นเกินไป (Noise)
    if len(cnt) < 15: continue
    
    points = cnt.reshape(-1, 2)
    
    # --- คำนวณจุดเริ่มต้นของเส้นในหน่วยเมตร ---
    start_pixel = points[0]
    # แปลง pixel เป็นเมตร และบวก offset
    target_x = (start_pixel[0] * scale_factor) + offset_x
    # หมายเหตุ: แกน Y ของภาพมักจะกลับด้านกับแกน Y ของหุ่นยนต์ (ภาพ: ลงคือบวก, หุ่นยนต์: ขึ้นคือบวก)
    # จึงใช้การลบจาก offset_y เพื่อกลับด้าน
    target_y = offset_y - (start_pixel[1] * scale_factor) 
    
    # --- A. TRAVEL MOVE (เคลื่อนที่ในอากาศ) ---
    # จากจุดปัจจุบัน -> ไปยังจุดเหนือจุดเริ่มต้นวาด (ที่ความสูง Z_SAFE)
    target_pos_air = np.array([target_x, target_y, Z_SAFE])
    # segment_a = generate_segment(current_pos, target_pos_air, SPEED_TRAVEL, SAMPLING_TIME)
    # full_trajectory_points.append(segment_a)
    full_trajectory_points.append(current_pos)
    full_trajectory_points.append(target_pos_air)
    current_pos = target_pos_air # อัปเดตตำแหน่งปัจจุบัน
    
    # --- B. PEN DOWN (กดปากกาลง) ---
    # จาก Z_SAFE -> ลงไปที่ Z_DRAW (ที่ตำแหน่ง X,Y เดิม)
    target_pos_down = np.array([target_x, target_y, Z_DRAW])
    segment_b = generate_segment(current_pos, target_pos_down, SPEED_TRAVEL, SAMPLING_TIME)
    full_trajectory_points.append(segment_b)
    current_pos = target_pos_down

    # --- C. DRAWING MOVE (ลากเส้นบนพื้นผิว) ---
    # วนลูปจุดที่เหลือในเส้นนั้น (เริ่มจากจุดที่ 1 เพราะจุดที่ 0 คือจุดที่กดปากกาลงไปแล้ว)
    for p in points[1:]: 
        px = (p[0] * scale_factor) + offset_x
        py = offset_y - (p[1] * scale_factor) # กลับด้านแกน Y เช่นกัน
        draw_pos = np.array([px, py, Z_DRAW])
        
        # สร้าง segment ระหว่างจุดต่อจุดใน contour
        segment_c = generate_segment(current_pos, draw_pos, SPEED_DRAW, SAMPLING_TIME)
        full_trajectory_points.append(segment_c)
        current_pos = draw_pos # อัปเดตตำแหน่งปัจจุบันไปเรื่อยๆ ตามเส้น
        
    # --- D. PEN UP (ยกปากกาขึ้นเมื่อจบเส้น) ---
    # จาก Z_DRAW -> ขึ้นไปที่ Z_SAFE (ที่ตำแหน่ง X,Y สุดท้ายของเส้น)
    target_pos_up = np.array([current_pos[0], current_pos[1], Z_SAFE])
    segment_d = generate_segment(current_pos, target_pos_up, SPEED_TRAVEL, SAMPLING_TIME)
    full_trajectory_points.append(segment_d)
    current_pos = target_pos_up

# --- 4. รวมข้อมูลและบันทึก CSV ---
# รวม List ของ array ย่อยๆ ให้เป็น Matrix ใหญ่ (N, 3)
all_points = np.vstack(full_trajectory_points)

# สร้าง DataFrame
df = pd.DataFrame(all_points, columns=['x', 'y', 'z'])
# เพิ่มคอลัมน์เวลา (t) เพื่อใช้อ้างอิง
df['t'] = np.arange(len(df)) * SAMPLING_TIME

print(f"Generated {len(df)} trajectory points.")
# บันทึกลง CSV (เผื่อนำไปใช้ต่อ)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved trajectory to {OUTPUT_CSV}")

# --- 5. Visualization 3D ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# พลอตจุดทั้งหมด
# ใช้เทคนิคการใส่สี (cmap) ตามความสูง (Z)
# สีโทนแดง = อยู่สูง (Travel), สีโทนน้ำเงิน = อยู่ต่ำ (Drawing)
sc = ax.scatter(df['x'], df['y'], df['z'], c=df['z'], cmap='coolwarm_r', s=1, label='Trajectory Points')

# ตกแต่งกราฟ
ax.set_title(f"Generated Robot Trajectory (dt={SAMPLING_TIME}s)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")

# เพิ่ม Colorbar เพื่อบอกระดับความสูง
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Z Height (m)')

# ตั้งค่าให้สเกลแกน X และ Y เท่ากัน เพื่อให้รูปไม่เบี้ยว
# หาจุดกึ่งกลางและระยะที่ไกลที่สุดเพื่อสร้าง bounding box
mid_x = (df['x'].max() + df['x'].min()) * 0.5
mid_y = (df['y'].max() + df['y'].min()) * 0.5
max_range = np.array([df['x'].max()-df['x'].min(), df['y'].max()-df['y'].min()]).max() / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
# แกน Z fix ไว้ให้เห็นชัดๆ
ax.set_zlim(Z_DRAW - 0.01, Z_SAFE + 0.02)

# ปรับมุมมองเริ่มต้น (optional)
ax.view_init(elev=30, azim=-60)

plt.show()