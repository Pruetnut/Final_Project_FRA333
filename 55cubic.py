import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_Cubic.csv"

# Robot Settings
DT = 0.008  # Sampling time

# Target Speeds
SPEED_TRAVEL = 0.25
SPEED_DRAW   = 0.05

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER: CUBIC SPLINE GENERATOR ---

def generate_cubic_trajectory(points, target_speed, dt):
    """
    สร้าง Trajectory ด้วย CubicSpline Interpolation
    วิ่งชนทุกจุด (Exact fit) แต่ควบคุมความเร็วให้คงที่
    """
    points = np.array(points)
    
    # 1. กรองจุดซ้ำ (CubicSpline ห้ามมีจุดซ้อนกันเด็ดขาด)
    # คำนวณระยะระหว่างจุด
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # เก็บจุดแรก + จุดที่ระยะห่าง > 0
    valid_mask = np.hstack(([True], dists > 1e-6))
    points = points[valid_mask]
    
    if len(points) < 2:
        return None, None, None

    # 2. Parameterize by Distance (คำนวณเวลาที่แต่ละจุด based on ระยะทาง)
    # ระยะทางสะสม (Cumulative Distance)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    # เวลาที่จุด Waypoint (Time = Distance / Speed)
    t_waypoints = cum_dist / target_speed
    total_time = t_waypoints[-1]
    
    # 3. สร้าง Cubic Spline Object
    # bc_type='clamped' คือบังคับให้ความเร็วต้นและปลาย = 0 (หยุดหัวท้าย)
    # axis=0 คือทำแยก xyz
    cs = CubicSpline(t_waypoints, points, axis=0, bc_type='clamped')
    
    # 4. Resample (สร้างจุดใหม่ตามเวลา Sampling Time)
    t_eval = np.arange(0, total_time, dt)
    
    # Evaluate Position
    pos = cs(t_eval)
    
    # Evaluate Velocity (Diff ครั้งที่ 1)
    vel = cs(t_eval, nu=1)
    
    # Evaluate Acceleration (Diff ครั้งที่ 2)
    acc = cs(t_eval, nu=2)
    
    return pos, vel, acc

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    
    # LOGIC: แบ่ง Chunk เหมือนเดิม
    if waypoints[i]['type'] == 1 and waypoints[i+1]['type'] == 1:
        # === DRAWING CHUNK ===
        chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        k = i + 1
        while k < len(waypoints) and waypoints[k]['type'] == 1:
            chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            k += 1
            
        pos, vel, acc = generate_cubic_trajectory(chunk, SPEED_DRAW, DT)
        i = k - 1
        seg_type = 1
        
    else:
        # === TRAVEL CHUNK ===
        chunk = [
            [p_curr['x'], p_curr['y'], p_curr['z']],
            [waypoints[i+1]['x'], waypoints[i+1]['y'], waypoints[i+1]['z']]
        ]
        pos, vel, acc = generate_cubic_trajectory(chunk, SPEED_TRAVEL, DT)
        i += 1
        seg_type = 0

    # บันทึกข้อมูล
    if pos is not None:
        for j in range(len(pos)):
            full_traj_rows.append({
                't': global_t,
                'x': pos[j,0], 'y': pos[j,1], 'z': pos[j,2],
                'vx': vel[j,0], 'vy': vel[j,1], 'vz': vel[j,2],
                'ax': acc[j,0], 'ay': acc[j,1], 'az': acc[j,2],
                'type': seg_type
            })
            global_t += DT

# --- 5. SAVE & PLOT ---
df_final = pd.DataFrame(full_traj_rows)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Saved {len(df_final)} points.")

# === VISUALIZATION ===
fig = plt.figure(figsize=(12, 6))

# Plot 1: 3D Path
ax1 = fig.add_subplot(121, projection='3d')
plot_df = df_final.iloc[::5]
colors = ['green' if t == 1 else 'red' for t in plot_df['type']]
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1)
ax1.set_title("Cubic Spline Path (Exact Fit)")

# Plot 2: Speed Profile
ax2 = fig.add_subplot(122)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], speed, label='Speed', color='black', linewidth=1)

# Target Line
ax2.axhline(y=SPEED_DRAW, color='green', linestyle='--', alpha=0.5, label='Target Draw')

ax2.set_title("Velocity Profile (Cubic)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (m/s)")
ax2.legend()
ax2.grid(True)

plt.show()