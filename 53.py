import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_SinglePath.csv"

# Robot Settings
DT = 0.008  # Sampling time (8ms)

# --- SPEED SETTINGS (กำหนดความเร็วตรงนี้) ---
SPEED_TRAVEL = 0.25  # m/s (ความเร็วตอนยก/เคลื่อนย้าย) - เร็ว
SPEED_DRAW   = 0.05  # m/s (ความเร็วตอนจิ้มวาด) - ช้าเพื่อให้แม่น

# ความนุ่มนวล
ACCEL_TIME = 0.1     # เวลาที่ใช้เร่งความเร็ว (s)
BLEND_RADIUS = 0.005 # รัศมีตีโค้ง (5mm)

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)

# แปลง type ให้เป็นตัวเลขแน่นอน (จัดการ 'HOME' หรือ string อื่นๆ ให้เป็น 0)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')

print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. PREPARE SINGLE PATH DATA ---
# เราจะรวบทุกจุดเป็น list เดียว แต่จะคำนวณ "เวลา" ของแต่ละช่วงแยกกัน

via_points = []
segment_times = [] # เก็บเวลาที่ยอมให้ใช้ในแต่ละช่วง

# ดึงจุดแรก
via_points.append([waypoints[0]['x'], waypoints[0]['y'], waypoints[0]['z']])

print("Calculating time duration for each segment...")

for i in range(len(waypoints) - 1):
    curr_wp = waypoints[i]
    next_wp = waypoints[i+1]
    
    p1 = np.array([curr_wp['x'], curr_wp['y'], curr_wp['z']])
    p2 = np.array([next_wp['x'], next_wp['y'], next_wp['z']])
    
    # เก็บจุดปลายทาง
    via_points.append(p2)
    
    # คำนวณระยะทาง
    dist = np.linalg.norm(p2 - p1)
    
    # ถ้าจุดซ้ำกัน (ระยะเป็น 0) ให้ข้ามไป (ใส่เวลาน้อยมากๆ เพื่อกัน error)
    if dist < 1e-6:
        segment_times.append(0.01)
        continue

    # --- LOGIC กำหนดความเร็ว ---
    # ถ้าจุดปลายทางเป็น Type 1 (Draw) แสดงว่าช่วงนี้คือการวาด -> ใช้ความเร็วช้า
    # ถ้าจุดปลายทางเป็น Type 0 (Travel) แสดงว่าช่วงนี้คือการยก -> ใช้ความเร็วเร็ว
    if next_wp['type'] == 1:
        target_speed = SPEED_DRAW
    else:
        target_speed = SPEED_TRAVEL
        
    # คำนวณเวลาที่ต้องใช้ (Time = Distance / Speed)
    duration = dist / target_speed
    
    # ป้องกันเวลาสั้นเกินไปจน Robot ทำไม่ทัน (Limit ขั้นต่ำไว้ที่ 0.016s หรือ 2 ticks)
    duration = max(duration, 0.016)
    
    segment_times.append(duration)

# แปลงเป็น Numpy Array
via_points = np.array(via_points)
segment_times = np.array(segment_times)

print(f"Total Segments: {len(segment_times)}")
print(f"Total Estimated Time: {np.sum(segment_times):.2f} seconds")

# --- 4. GENERATE TRAJECTORY (SINGLE SHOT) ---
print("Computing Global Trajectory (mstraj)...")

# rtb.mstraj คือพระเอกของงานนี้
# dt: คือ array บอกเวลาของแต่ละช่วง (ไม่ใช่ sampling time)
# tacc: เวลาเร่ง
# qdmax: ไม่ต้องใส่ เพราะเรากำหนดผ่าน dt (segment_times) แล้ว
traj = rtb.mstraj(via_points, dt=DT, tacc=ACCEL_TIME, qdmax=SPEED_DRAW)

# ดึงข้อมูล
pos = traj.q
# คำนวณ v, a ด้วยการ diff
vel = np.gradient(pos, DT, axis=0)
acc = np.gradient(vel, DT, axis=0)

# --- 5. MAPPING TYPE BACK (Map ค่า Type กลับมาใส่ใน Trajectory) ---
# เนื่องจาก mstraj มันรวมทุกอย่างเป็นเส้นเดียว เราต้อง map type กลับมาเพื่อให้รู้ว่าช่วงไหนคืออะไร
# วิธีแบบง่าย: ใช้ KDTree หาว่าจุด trajectory นี้ ใกล้ waypoint ไหนที่สุด
from scipy.spatial import cKDTree

# สร้าง Tree จาก Waypoint เดิม
waypoint_coords = df[['x', 'y', 'z']].values
tree = cKDTree(waypoint_coords)

# หา Type ของแต่ละจุดใน Trajectory
print("Mapping types back to trajectory...")
_, indices = tree.query(pos) # หา index ของ waypoint ที่ใกล้ที่สุด
traj_types = df.iloc[indices]['type'].values

# --- 6. SAVE & PLOT ---
# สร้าง DataFrame ผลลัพธ์
df_final = pd.DataFrame({
    't': np.arange(len(pos)) * DT,
    'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2],
    'vx': vel[:, 0], 'vy': vel[:, 1], 'vz': vel[:, 2],
    'ax': acc[:, 0], 'ay': acc[:, 1], 'az': acc[:, 2],
    'type': traj_types
})

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Saved {len(df_final)} points to {OUTPUT_CSV}")

# === VISUALIZATION ===
fig = plt.figure(figsize=(12, 6))

# Plot 1: 3D Path
ax1 = fig.add_subplot(121, projection='3d')
plot_df = df_final.iloc[::5] # Downsample

# กำหนดสี: Type 1=Draw(เขียว), Type 0=Travel(แดง)
colors = ['green' if t == 1 else 'red' for t in plot_df['type']]
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1)
ax1.set_title("Single Continuous Path (mstraj)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# Plot 2: Speed Profile
ax2 = fig.add_subplot(122)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], speed, label='Speed', color='black', linewidth=1)

# Highlight Drawing Areas
mask = df_final['type'] == 1
ax2.fill_between(df_final['t'], 0, np.max(speed), where=mask, color='green', alpha=0.2, label='Drawing')

# ขีดเส้นความเร็วเป้าหมายให้ดู
ax2.axhline(y=SPEED_DRAW, color='green', linestyle='--', alpha=0.5, label='Target Draw Speed')
ax2.axhline(y=SPEED_TRAVEL, color='red', linestyle='--', alpha=0.5, label='Target Travel Speed')

ax2.set_title("Velocity Profile (Should be continuous)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (m/s)")
ax2.legend()
ax2.grid(True)

plt.show()