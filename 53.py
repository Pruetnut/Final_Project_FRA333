import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_SmoothSpline.csv"

# Robot Settings
DT = 0.02  # Sampling time (8ms)

# Target Speeds
SPEED_TRAVEL = 0.25 # m/s (ตอนยก)
SPEED_DRAW   = 0.05 # m/s (ตอนวาด - เอาช้าๆ ให้ชัวร์)

# *** ตัวแปรสำคัญที่สุด ***
# 0.0 = ผ่านทุกจุดเป๊ะๆ (แต่อาจจะสวิงบ้างถ้าจุดไม่เนียน)
# 0.0001 - 0.001 = ยอมให้คลาดเคลื่อนนิดเดียว เพื่อแลกกับความสมูท (แนะนำ!)
SMOOTHING_FACTOR = 0.0005 

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')

print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTIONS ---

def generate_spline_trajectory(points, target_speed, dt, s_factor=0):
    """
    สร้าง Trajectory ด้วย B-Spline (รับประกันความสมูท)
    """
    points = np.array(points)
    
    # 1. Cleaning: ลบจุดซ้ำ (Spline ไม่ชอบจุดทับกัน)
    # เช็คระยะห่างระหว่างจุด
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # เก็บจุดแรก + จุดที่มีระยะห่าง > 0
    valid_idx = np.hstack(([True], diff > 1e-6))
    points = points[valid_idx]

    # ถ้าจุดน้อยเกินไป ทำ Spline ไม่ได้ (ต้องมีอย่างน้อย 4 จุดสำหรับ cubic spline)
    if len(points) < 4:
        # Fallback: ใช้ Linear Interpolation ธรรมดา
        total_dist = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        duration = max(total_dist / target_speed, 2 * dt)
        num_steps = int(duration / dt)
        t_eval = np.linspace(0, 1, num_steps)
        
        # Linear interp
        pos = np.zeros((num_steps, 3))
        for i in range(3):
            pos[:, i] = np.interp(t_eval, np.linspace(0, 1, len(points)), points[:, i])
        
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        return pos, vel, acc

    # 2. คำนวณระยะทางรวม เพื่อหาเวลาที่ต้องใช้ (Duration)
    # เพื่อคุมความเร็วให้ได้ตาม target_speed
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_len = np.sum(dists)
    total_time = total_len / target_speed
    
    # จำนวน Sampling points
    num_samples = int(total_time / dt)
    if num_samples < 2: num_samples = 2
    
    # 3. สร้าง B-Spline (Curve Fitting)
    # tck คือ parameters ของเส้นโค้ง, u คือตำแหน่งบนเส้น (0 ถึง 1)
    # s (smoothing) คือตัวช่วยลดการสวิง!
    try:
        tck, u = splprep(points.T, s=s_factor, k=3) # k=3 means Cubic Spline (C2 continuous)
    except Exception as e:
        print(f"Spline Error: {e}, using raw points")
        return points, np.zeros_like(points), np.zeros_like(points)

    # 4. Evaluate (สุ่มจุดบนเส้นโค้งออกมาตามเวลา)
    u_new = np.linspace(0, 1, num_samples)
    
    # คำนวณ Position
    x, y, z = splev(u_new, tck)
    pos = np.vstack((x, y, z)).T
    
    # คำนวณ Velocity (Derivative 1)
    # หมายเหตุ: splev ให้ diff เทียบกับ u (0->1) ต้องแปลงเป็นเทียบเวลา t (0->TotalTime)
    # v_real = v_spline * (du/dt) = v_spline * (1/TotalTime)
    x_d1, y_d1, z_d1 = splev(u_new, tck, der=1)
    vel = np.vstack((x_d1, y_d1, z_d1)).T * (1.0 / total_time)
    
    # คำนวณ Acceleration (Derivative 2)
    # a_real = a_spline * (1/TotalTime)^2
    x_d2, y_d2, z_d2 = splev(u_new, tck, der=2)
    acc = np.vstack((x_d2, y_d2, z_d2)).T * (1.0 / total_time**2)
    
    return pos, vel, acc

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    
    # LOGIC: แบ่ง Chunk เหมือนเดิม (Travel vs Draw)
    if waypoints[i]['type'] == 1 and waypoints[i+1]['type'] == 1:
        # === DRAWING CHUNK (Spline Mode) ===
        draw_chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        k = i + 1
        while k < len(waypoints) and waypoints[k]['type'] == 1:
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            k += 1
            
        # ใช้ Spline สร้างเส้นโค้งที่สมูทที่สุด
        pos, vel, acc = generate_spline_trajectory(draw_chunk, SPEED_DRAW, DT, s_factor=SMOOTHING_FACTOR)
        
        i = k - 1
        seg_type = 1
        
    else:
        # === TRAVEL CHUNK (Linear/Spline Mode) ===
        # ช่วงเดินทางสั้นๆ ใช้ Spline แบบเส้นตรง (k=1) หรือ Spline ปกติก็ได้
        # แต่เพื่อความง่าย ใช้ฟังก์ชันเดียวกันแต่จุดน้อย
        p_next = waypoints[i+1]
        travel_chunk = [
            [p_curr['x'], p_curr['y'], p_curr['z']],
            [p_next['x'], p_next['y'], p_next['z']]
        ]
        
        pos, vel, acc = generate_spline_trajectory(travel_chunk, SPEED_TRAVEL, DT, s_factor=0)
        
        i += 1
        seg_type = 0

    # Record Data
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
print(f"Saved {len(df_final)} points to {OUTPUT_CSV}")

# === VISUALIZATION ===
fig = plt.figure(figsize=(12, 6))

# Plot 1: 3D Path
ax1 = fig.add_subplot(121, projection='3d')
plot_df = df_final.iloc[::5]
colors = ['green' if t == 1 else 'red' for t in plot_df['type']]
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1)
ax1.set_title("Spline Path (Ideally Smooth)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# Plot 2: Velocity Profile
ax2 = fig.add_subplot(122)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], speed, label='Actual Speed', color='black', linewidth=1)

# Target Speed Line
ax2.plot(df_final['t'], [SPEED_DRAW if t==1 else SPEED_TRAVEL for t in df_final['type']], 
         color='orange', linestyle='--', alpha=0.5, label='Target Speed')

ax2.set_title("Velocity Profile (Spline-based)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (m/s)")
ax2.legend()
ax2.grid(True)

plt.show()