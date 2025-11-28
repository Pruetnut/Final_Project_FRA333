import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_GlobalOnePath.csv"

# Robot Settings
DT = 0.008  # Sampling time

# Target Speeds (เป้าหมายความเร็วเฉลี่ยของแต่ละช่วง)
SPEED_TRAVEL = 0.25
SPEED_DRAW   = 0.05

# Tuning
TIME_SAFETY_FACTOR = 5 # เผื่อเวลาเพิ่ม 10% กันกราฟสวิงเกินลิมิต

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)

# กรองจุดซ้ำออก (สำคัญมากสำหรับ Cubic Spline)
# ถ้าจุดซ้ำกัน เวลาจะเป็น 0 ทำให้คำนวณไม่ได้
df['dist_next'] = np.linalg.norm(df[['x','y','z']].diff().fillna(1).values, axis=1)
df = df[df['dist_next'] > 1e-6].reset_index(drop=True)

waypoints = df.to_dict('records')
points = df[['x', 'y', 'z']].values

print(f"Loaded {len(waypoints)} unique waypoints.")

# --- 3. CALCULATE TIME STAMPS (กำหนดเวลาให้แต่ละจุด) ---
# เราต้องสร้าง Time Vector [t0, t1, t2, ..., tn]
# โดย t0 = 0, tn = เวลารวมทั้งหมด

timestamps = [0.0]
current_time = 0.0

for i in range(len(waypoints) - 1):
    p_curr = np.array([waypoints[i]['x'], waypoints[i]['y'], waypoints[i]['z']])
    p_next = np.array([waypoints[i+1]['x'], waypoints[i+1]['y'], waypoints[i+1]['z']])
    
    dist = np.linalg.norm(p_next - p_curr)
    
    # เช็คว่าช่วงนี้เป็นช่วงอะไร (ดูที่จุดปลายทาง)
    # ถ้าจุดถัดไปเป็น Type 1 (Draw) -> ใช้ความเร็ววาด
    if waypoints[i+1]['type'] == 1:
        target_speed = SPEED_DRAW
    else:
        target_speed = SPEED_TRAVEL
        
    # คำนวณเวลาที่ควรใช้ (Time = Dist / Speed)
    duration = (dist / target_speed) * TIME_SAFETY_FACTOR
    
    # ป้องกันเวลาสั้นเกินไป (Minimum duration per segment)
    duration = max(duration, 0.016) 
    
    current_time += duration
    timestamps.append(current_time)

timestamps = np.array(timestamps)
total_time = timestamps[-1]

print(f"Total Path Time: {total_time:.2f} seconds")

# --- 4. GENERATE GLOBAL SPLINE ---
# bc_type='clamped' คือคำสั่งศักดิ์สิทธิ์ที่สั่งว่า
# "Velocity ที่จุดแรก และจุดสุดท้าย ต้องเท่ากับ 0" ((0,0,0), (0,0,0))
cs = CubicSpline(timestamps, points, axis=0, bc_type='clamped')

# --- 5. RESAMPLE (สร้างจุดตาม Sampling Time) ---
t_eval = np.arange(0, total_time, DT)

pos = cs(t_eval)        # Position
vel = cs(t_eval, nu=1)  # Velocity (Diff 1)
acc = cs(t_eval, nu=2)  # Acceleration (Diff 2)

# Map Type กลับมา (เพื่อความสวยงามตอนพลอต)
# หาว่าเวลา t_eval ช่วงนี้ อยู่ระหว่าง waypoint ไหน
# searchsorted จะบอก index ของ timestamp
indices = np.searchsorted(timestamps, t_eval) - 1
indices = np.clip(indices, 0, len(waypoints)-1)
traj_types = df.iloc[indices]['type'].values

# --- 6. SAVE & PLOT ---
df_final = pd.DataFrame({
    't': t_eval,
    'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2],
    'vx': vel[:, 0], 'vy': vel[:, 1], 'vz': vel[:, 2],
    'ax': acc[:, 0], 'ay': acc[:, 1], 'az': acc[:, 2],
    'type': traj_types
})

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df_final)} points to {OUTPUT_CSV}")

# === VISUALIZATION ===
fig = plt.figure(figsize=(12, 6))

# Plot 1: 3D Path
ax1 = fig.add_subplot(121, projection='3d')
plot_df = df_final.iloc[::5]
colors = ['green' if t == 1 else 'red' for t in plot_df['type']]
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1)
ax1.set_title("Single Global Cubic Spline")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# Plot 2: Velocity Profile
ax2 = fig.add_subplot(122)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], speed, label='Speed', color='black', linewidth=1.5)

# Highlight Drawing
mask = df_final['type'] == 1
ax2.fill_between(df_final['t'], 0, np.max(speed), where=mask, color='green', alpha=0.2, label='Drawing Phase')

# Reference Lines
ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Zero Speed')
ax2.axhline(y=SPEED_DRAW, color='green', linestyle='--', alpha=0.5, label='Target Draw')
ax2.axhline(y=SPEED_TRAVEL, color='blue', linestyle='--', alpha=0.5, label='Target Travel')

ax2.set_title("Velocity Profile (Continuous Flow)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (m/s)")
ax2.legend()
ax2.grid(True)

plt.show()