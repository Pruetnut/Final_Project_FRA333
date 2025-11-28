import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rucking import Rucking, InputParameter, OutputParameter, Result

# --- CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png" # เปลี่ยน path รูปของคุณ
OUTPUT_CSV = "To_matlab.csv"

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

# Ruckig Limits (กำหนดความเร็ว/เร่ง/กระชาก สูงสุดที่นี่)
# หน่วย: m/s, m/s^2, m/s^3
MAX_VEL = [0.25, 0.25, 0.25]  # [x, y, z]
MAX_ACC = [0.5, 0.5, 0.5]     # เร่งได้เร็ว
MAX_JERK = [3.0, 3.0, 3.0]    # ค่า Jerk สูง = ตอบสนองไว, ต่ำ = นุ่มนวลมาก

CONTROL_CYCLE = 0.008 # 8ms for UR5

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
    bilateral = cv2.bilateralFilter(blur,15,75,75)
    edges = cv2.Canny(bilateral, 50, 150)
    # ใช้ RETR_EXTERNAL เพื่อลดความซับซ้อน (เอาแค่เส้นนอก)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, new_h, w
# --- MAIN PIPELINE ---

# 1. Image Processing
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # ใช้ EXTERNAL เพื่อเอาแค่เส้นนอก หรือ LIST เอาหมด

# 2. Prepare Scaling
contours, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
scale_factor = CANVAS_WIDTH_M / img_w
# ตั้งจุด Origin (0,0) ของหุ่นยนต์ให้เริ่มที่มุมซ้ายล่างของรูป หรือตรงกลางตามต้องการ
offset_x = (-CANVAS_WIDTH_M) / 2 + OFFSET_X
offset_y = ((img_h * scale_factor) / 2) + OFFSET_Y
print(f"Detected {len(contours)} contours.")

# --- 3. BUILD KEY WAYPOINTS (สร้างจุดเช็คพอยท์หลัก) ---
# เราจะไม่ interpolate เอง แต่จะเก็บแค่จุดสำคัญ แล้วให้ Ruckig เชื่อมให้
waypoints_queue = []
# 3.0 Start at Home
waypoints_queue.append({'pos': [offset_x, offset_y, Z_SAFE], 'type': 'HOME'})

for cnt in contours:
    if len(cnt) < 15: continue # กรอง Noise
    
    points = cnt.reshape(-1, 2)
    
    # แปลงพิกัดทั้งหมดใน contour นี้เตรียมไว้
    real_points = []
    for p in points:
        px = (p[0] * scale_factor) + offset_x
        py = offset_y - (p[1] * scale_factor)
        real_points.append([px, py])
    
    # 3.1 TRAVEL: ยกปากกาไปหาจุดเริ่ม (Z_SAFE)
    start_x, start_y = real_points[0]
    waypoints_queue.append({'pos': [start_x, start_y, Z_SAFE], 'type': 'TRAVEL'})
    
    # 3.2 PEN DOWN: กดปากกาลง (Z_DRAW)
    waypoints_queue.append({'pos': [start_x, start_y, Z_DRAW], 'type': 'DOWN'})
    
    # 3.3 DRAWING: ลากเส้นตามจุดต่างๆ
    # เทคนิค: ถ้าจุดถี่มาก เราใส่ทุกจุดเลยก็ได้ Ruckig จะพยายามวิ่งผ่านให้
    for i in range(1, len(real_points)):
        rx, ry = real_points[i]
        waypoints_queue.append({'pos': [rx, ry, Z_DRAW], 'type': 'DRAW'})
        
    # 3.4 PEN UP: ยกปากกาขึ้นที่จุดสุดท้าย
    last_x, last_y = real_points[-1]
    waypoints_queue.append({'pos': [last_x, last_y, Z_SAFE], 'type': 'UP'})

# --- 4. RUCKIG TRAJECTORY GENERATION (The Core) ---

# Initialize Ruckig
otg = Ruckig(3, CONTROL_CYCLE) # 3 DOFs
inp = InputParameter(3)
out = OutputParameter(3)

# Config Limits
inp.max_velocity = MAX_VEL
inp.max_acceleration = MAX_ACC
inp.max_jerk = MAX_JERK

# Initial State
inp.current_position = [0.0, 0.0, Z_SAFE]
inp.current_velocity = [0.0, 0.0, 0.0]
inp.current_acceleration = [0.0, 0.0, 0.0]

full_trajectory_data = [] # เก็บผลลัพธ์ละเอียด (t, x, y, z, v, a)
total_time = 0.0

print(f"Generating smooth trajectory from {len(waypoints_queue)} key waypoints...")

for wp_idx, wp in enumerate(waypoints_queue):
    # กำหนดเป้าหมายใหม่
    inp.target_position = wp['pos']
    
    # Note: เราตั้ง Target Vel/Acc เป็น 0 เพื่อให้มั่นใจว่าหยุดนิ่งได้ถ้าจำเป็น (Robust)
    # แต่เนื่องจากจุด Drawing มันใกล้กันมาก Ruckig จะไม่เบรกจนหยุดสนิทจริงๆ แต่มันจะคำนวณ Motion ที่เหมาะสมให้
    inp.target_velocity = [0.0, 0.0, 0.0] 
    inp.target_acceleration = [0.0, 0.0, 0.0]
    
    # Loop จนกว่าจะถึงจุดเป้าหมาย (Waypoint นี้)
    while True:
        # คำนวณ Next Step
        result = otg.update(inp, out)
        
        # บันทึกข้อมูล
        full_trajectory_data.append([
            total_time,
            out.new_position[0], out.new_position[1], out.new_position[2],     # Pos
            out.new_velocity[0], out.new_velocity[1], out.new_velocity[2],     # Vel
            out.new_acceleration[0], out.new_acceleration[1], out.new_acceleration[2], # Acc
            wp['type'] # เก็บ Tag ไว้ดูว่าช่วงนี้ทำอะไร
        ])
        
        # อัพเดทค่า Input สำหรับรอบถัดไป
        out.pass_to_input(inp)
        total_time += CONTROL_CYCLE
        
        # เช็คว่าจบ Segment นี้หรือยัง
        if result == Result.Finished:
            break

# --- 5. CREATE DATAFRAME & VISUALIZE ---
columns = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'type']
df_final = pd.DataFrame(full_trajectory_data, columns=columns)

print(f"Generation Complete!")
print(f"Total Time: {total_time:.2f} s")
print(f"Total Points: {len(df_final)}")
print(df_final.head())

# Save to CSV
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")

# Plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot โดยใช้สีตามความเร็วแกน Z (สีแดง=กำลังยก/ลง, สีน้ำเงิน=นิ่ง/วาด)
sc = ax.scatter(df_final['x'][::10], df_final['y'][::10], df_final['z'][::10], 
                c=df_final['vz'][::10], cmap='coolwarm', s=1)
ax.set_title("Ruckig Generated Trajectory (Jerk Limited)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Plot Velocity Profile (2D) เพื่อดูความสมูท
plt.figure(figsize=(10,5))
plt.plot(df_final['t'], df_final['vx'], label='Vx')
plt.plot(df_final['t'], df_final['vy'], label='Vy')
plt.plot(df_final['t'], df_final['vz'], label='Vz')
plt.title("Velocity Profile (Check Smoothness)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid(True)

plt.show()

# full_trajectory_points = []
# current_pos = np.array([offset_x, offset_y, Z_SAFE]) # เริ่มที่ Home
# ใส่จุด Home ลงไปก่อน
# full_trajectory_points.append(current_pos)

# for i, cnt in enumerate(contours):
#     # กรอง Noise เล็กๆ
#     if len(cnt) < 15: continue
    
#     points = cnt.reshape(-1, 2)
    
#     # --- A. TRAVEL MOVE (Air) ---
#     # จากจุดปัจจุบัน -> ไปจุดเริ่มของเส้น (ที่ความสูง Safe)
#     start_pixel = points[0]
#     target_x = (start_pixel[0] * scale_factor) + offset_x
#     target_y = offset_y - (start_pixel[1] * scale_factor) # กลับแกน Y
#     target_pos = np.array([target_x, target_y, Z_SAFE])
    
#     # สร้างจุดย่อย
#     segment = generate_segment(current_pos, target_pos, SPEED_TRAVEL, SAMPLING_TIME)
#     full_trajectory_points.append(segment)
#     current_pos = target_pos # อัพเดทตำแหน่งล่าสุด
    
#     # --- B. PEN DOWN (Z-Axis Move) ---
#     # จาก Safe -> Draw (ที่ตำแหน่ง X,Y เดิม)
#     down_pos = np.array([target_x, target_y, Z_DRAW])
#     segment = generate_segment(current_pos, down_pos, SPEED_TRAVEL * 0.5, SAMPLING_TIME) # ลงช้าๆหน่อย
#     full_trajectory_points.append(segment)
#     current_pos = down_pos

#     # --- C. DRAWING MOVE (Surface) ---
#     # วนลูปจุดทั้งหมดในเส้นนั้น
#     for p in points[1:]: # เริ่มจุดที่ 2
#         px = (p[0] * scale_factor) + offset_x
#         py = offset_y - (p[1] * scale_factor)
#         draw_pos = np.array([px, py, Z_DRAW])
        
#         segment = generate_segment(current_pos, draw_pos, SPEED_DRAW, SAMPLING_TIME)
#         full_trajectory_points.append(segment)
#         current_pos = draw_pos
        
#     # --- D. PEN UP (Z-Axis Move) ---
#     # จาก Draw -> Safe
#     up_pos = np.array([current_pos[0], current_pos[1], Z_SAFE])
#     segment = generate_segment(current_pos, up_pos, SPEED_TRAVEL * 0.5, SAMPLING_TIME)
#     full_trajectory_points.append(segment)
#     current_pos = up_pos



# รวม List ทั้งหมดเป็น Array ใหญ่ (N, 3)
# vstack จะรวม array ย่อยๆ ต่อกัน
# all_points = np.vstack(full_trajectory_points)

# # --- 4. Calculate Kinematics (v, a) ---
# # สร้าง DataFrame
# df_point = pd.DataFrame(all_points, columns=['x', 'y', 'z'])



# # Filter: ลบจุดแรกๆ ที่อาจจะมีค่ากระโดดจากการคำนวณ Gradient
# df = df_point.iloc[1:].reset_index(drop=True)

# print(f"Generated {len(df)} waypoints with {SAMPLING_TIME}s sampling time.")
# print(df[['x', 'y', 'z']].head())

# # Save CSV
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"Saved to {OUTPUT_CSV}")

# all_points.to_csv("all_point", index=False)
# print(f"Saved to all point")

# # --- 5. Visualization 3D ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # เพื่อความเร็วในการพลอต เราจะสุ่มพลอตแค่บางจุด (เช่น ทุกๆ 10 จุด) ถ้าข้อมูลเยอะ
# skip = 5
# ax.scatter(df['x'][::skip], df['y'][::skip], df['z'][::skip], c=df['z'][::skip], cmap='coolwarm', s=1)

# ax.set_title(f"Trajectory Simulation (Sampled every {SAMPLING_TIME}s)")
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")

# # Scale adjustment
# max_range = np.array([df['x'].max()-df['x'].min(), df['y'].max()-df['y'].min(), df['z'].max()-df['z'].min()]).max() / 2.0
# mid_x = (df['x'].max()+df['x'].min()) * 0.5
# mid_y = (df['y'].max()+df['y'].min()) * 0.5
# mid_z = (df['z'].max()+df['z'].min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# plt.show()