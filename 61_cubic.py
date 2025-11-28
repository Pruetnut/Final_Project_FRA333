import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_StrictLimit.csv"

# Robot Settings
DT = 0.008

# *** HARD LIMITS (รับประกันไม่เกินนี้) ***
LIMIT_TRAVEL = 0.5 
LIMIT_DRAW   = 0.1 

# Smoothing (ถ้ายังสวิง ให้เพิ่มค่านี้)
SMOOTHING_FACTOR = 0.0001

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTION (ITERATIVE SCALING) ---

def generate_strict_spline_segment(points, speed_limit, dt, s_factor):
    points = np.array(points)
    
    # Clean duplicates
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid_idx = np.hstack(([True], diff > 1e-6))
    points = points[valid_idx]

    # กรณีจุดน้อยเกินไป (Spline ไม่ไหว ให้ใช้ Linear)
    if len(points) < 4:
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_len = np.sum(dists)
        duration = max(total_len / speed_limit, 2 * dt)
        num_steps = int(duration / dt)
        t_eval = np.linspace(0, 1, num_steps)
        pos = np.zeros((num_steps, 3))
        for i in range(3):
            pos[:, i] = np.interp(t_eval, np.linspace(0, 1, len(points)), points[:, i])
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        return pos, vel, acc

    # --- STEP 1: เตรียม Spline ---
    try:
        # สร้าง Spline ครั้งเดียว
        tck, u = splprep(points.T, s=s_factor, k=3)
    except:
        return points, np.zeros_like(points), np.zeros_like(points)

    # ประมาณการเวลาเริ่มต้น
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    current_duration = np.sum(dists) / speed_limit
    
    # --- STEP 2: ITERATIVE LOOP (วนลูปแก้เวลา) ---
    max_iterations = 20 # กัน Loop ไม่จบ (แต่ปกติ 2-3 รอบก็จบ)
    
    for _ in range(max_iterations):
        # 2.1 สร้างข้อมูลตามเวลาปัจจุบัน
        num_samples = max(int(current_duration / dt), 2)
        u_new = np.linspace(0, 1, num_samples)
        
        # คำนวณความเร็ว
        der1 = splev(u_new, tck, der=1)
        vel = np.vstack(der1).T * (1.0 / current_duration)
        
        # หา Max Speed ที่เกิดขึ้นจริง
        speed_norm = np.linalg.norm(vel, axis=1)
        max_speed_actual = np.max(speed_norm)
        
        # 2.2 เช็คว่าเกิน Limit ไหม?
        if max_speed_actual <= speed_limit:
            # ผ่าน! ออกจากลูปได้เลย
            break
        else:
            # ไม่ผ่าน! คำนวณ Ratio การเกิน
            ratio = max_speed_actual / speed_limit
            
            # ขยายเวลาเพิ่มตาม Ratio + แถมให้อีก 5% (Safety Buffer) เพื่อให้จบไวๆ
            current_duration = current_duration * ratio * 1.05
            
    # --- STEP 3: FINAL CALCULATION ---
    # ใช้ current_duration ล่าสุดที่ผ่านเกณฑ์แล้ว
    final_num_samples = max(int(current_duration / dt), 2)
    u_final = np.linspace(0, 1, final_num_samples)
    
    # Position
    x, y, z = splev(u_final, tck)
    pos = np.vstack((x, y, z)).T
    
    # Velocity
    der1 = splev(u_final, tck, der=1)
    vel = np.vstack(der1).T * (1.0 / current_duration)
    
    # Acceleration
    der2 = splev(u_final, tck, der=2)
    acc = np.vstack(der2).T * (1.0 / current_duration**2)
    
    return pos, vel, acc

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    
    if waypoints[i]['type'] == 1 and waypoints[i+1]['type'] == 1:
        # DRAW CHUNK
        chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        k = i + 1
        while k < len(waypoints) and waypoints[k]['type'] == 1:
            chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            k += 1
            
        pos, vel, acc = generate_strict_spline_segment(chunk, LIMIT_DRAW, DT, SMOOTHING_FACTOR)
        i = k - 1
        seg_type = 1
        
    else:
        # TRAVEL CHUNK
        chunk = [
            [p_curr['x'], p_curr['y'], p_curr['z']],
            [waypoints[i+1]['x'], waypoints[i+1]['y'], waypoints[i+1]['z']]
        ]
        # Travel อาจไม่ต้อง strict มาก หรือใช้ strict ก็ได้
        pos, vel, acc = generate_strict_spline_segment(chunk, LIMIT_TRAVEL, DT, 0)
        i += 1
        seg_type = 0

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
print(f"Saved {len(df_final)} points.")

# VISUALIZATION
fig = plt.figure(figsize=(12, 6))
ax2 = fig.add_subplot(111)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], speed, label='Actual Speed', color='black', linewidth=1)

# Plot Limits
ax2.axhline(y=LIMIT_DRAW, color='green', linestyle='-', label='Limit Draw')
ax2.axhline(y=LIMIT_TRAVEL, color='red', linestyle='-', label='Limit Travel')

ax2.set_title("Velocity Profile (Strictly Limited)")
ax2.set_ylabel("Speed (m/s)")
ax2.legend()
ax2.grid(True)
plt.show()