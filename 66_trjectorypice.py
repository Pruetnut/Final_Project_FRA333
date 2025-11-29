import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_Fixed.csv"

# Robot Settings
DT = 0.008 

# Physics Limits
LIMIT_VEL_DRAW = 0.05
LIMIT_ACC_DRAW = 0.1
LIMIT_VEL_TRAVEL = 0.25
LIMIT_ACC_TRAVEL = 0.5

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found. Please run Step 1 again.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTIONS ---

def get_quintic_duration(dist, v_limit, a_limit):
    if dist < 1e-6: return 0.1
    t_vel = (1.875 * dist) / v_limit
    t_acc = np.sqrt((5.77 * dist) / a_limit)
    return max(t_vel, t_acc, 0.1)

def generate_quintic_travel(p_start, p_end, v_limit, a_limit, dt):
    """ใช้สำหรับ Travel หรือเส้นวาดที่สั้นมากๆ"""
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    dist = np.linalg.norm(p_end - p_start)
    
    if dist < 1e-6: return None, None, None # จุดซ้ำจริงๆ ข้ามได้

    duration = get_quintic_duration(dist, v_limit, a_limit)
    num_steps = max(int(np.ceil(duration / dt)), 2)
    
    t = np.linspace(0, 1, num_steps)
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    dds = 60*t - 180*t**2 + 120*t**3
    
    pos = p_start + (p_end - p_start) * s[:, np.newaxis]
    vel = (p_end - p_start) * ds[:, np.newaxis] / duration
    acc = (p_end - p_start) * dds[:, np.newaxis] / (duration**2)
    
    return pos, vel, acc

def generate_strict_continuous_draw(points, v_limit, a_limit, dt):
    """Draw Path: พยายามใช้ Spline แต่ถ้าไม่ได้ให้ใช้ Quintic (Fallback)"""
    points = np.array(points)
    
    # 1. CLEANING: ผ่อนปรนเงื่อนไขลง (1e-6)
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid_mask = np.hstack(([True], diff > 1e-6)) 
    points = points[valid_mask]
    
    # *** FALLBACK LOGIC ***
    # ถ้าจุดเหลือน้อยกว่า 3 จุด -> ทำ Spline ไม่ได้ -> ให้ลากเส้นตรง (Quintic) เชื่อมหัวท้ายเลย
    if len(points) < 3:
        # print("Points too few for Spline, using Quintic fallback.")
        return generate_quintic_travel(points[0], points[-1], v_limit, a_limit, dt)

    # 2. FLATTEN Z
    z_target = points[0, 2] 
    points[:, 2] = z_target

    # 3. Iterative Time Scaling
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_dist = np.sum(dists)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    current_duration = total_dist / v_limit * 1.5 # เผื่อเวลาเริ่มต้นเยอะหน่อย (1.5x)
    t_points_norm = cum_dist / total_dist
    
    success = False
    
    for _ in range(20):
        t_points_real = t_points_norm * current_duration
        try:
            cs = CubicSpline(t_points_real, points, axis=0, bc_type='clamped')
        except:
            # ถ้า Spline Error ให้ Break ไปใช้ Fallback
            break
            
        num_steps = int(np.ceil(current_duration / dt))
        if num_steps < 2: num_steps = 2
        t_eval = np.linspace(0, current_duration, num_steps)
        
        acc_check = cs(t_eval, nu=2)
        a_peak = np.max(np.linalg.norm(acc_check, axis=1))
        
        if a_peak <= a_limit:
            success = True
            break
        else:
            ratio = np.sqrt(a_peak / a_limit)
            current_duration = current_duration * ratio * 1.05

    if not success:
        # ถ้าพยายามแล้วยัง Error หรือไม่ผ่าน Limit -> ใช้ Quintic เชื่อมหัวท้าย (ปลอดภัยสุด)
        # print("Spline optimization failed, using Quintic fallback.")
        return generate_quintic_travel(points[0], points[-1], v_limit, a_limit, dt)

    # Final Generate
    final_steps = max(int(np.ceil(current_duration / dt)), 2)
    t_final = np.linspace(0, current_duration, final_steps)
    
    pos = cs(t_final)
    vel = cs(t_final, nu=1)
    acc = cs(t_final, nu=2)
    
    return pos, vel, acc

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    p_next = waypoints[i+1]
    
    # LOGIC CHUNKING
    is_drawing_transition = (p_curr['type'] == 0 and p_next['type'] == 1)
    is_drawing_continuous = (p_curr['type'] == 1 and p_next['type'] == 1)
    
    if is_drawing_transition or is_drawing_continuous:
        # === DRAW CHUNK ===
        draw_chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        k = i + 1
        while k < len(waypoints):
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            if waypoints[k]['type'] == 0: 
                break 
            k += 1
            
        # เรียกฟังก์ชันที่มี Fallback แล้ว
        pos, vel, acc = generate_strict_continuous_draw(draw_chunk, LIMIT_VEL_DRAW, LIMIT_ACC_DRAW, DT)
        
        # ถ้า pos เป็น None (กรณีซ้ำจริงๆ) ให้ข้าม
        if pos is None:
             # กรณีเลวร้ายสุดๆ ให้ทำ Travel ข้ามไปเลย
            i = k - 1
            continue

        i = k - 1 
        segment_type = 1
        
    else:
        # === TRAVEL CHUNK ===
        start_pt = [p_curr['x'], p_curr['y'], p_curr['z']]
        end_pt   = [p_next['x'], p_next['y'], p_next['z']]
        
        pos, vel, acc = generate_quintic_travel(start_pt, end_pt, LIMIT_VEL_TRAVEL, LIMIT_ACC_TRAVEL, DT)
        
        i += 1
        segment_type = 0

    if pos is not None:
        for j in range(len(pos)):
            full_traj_rows.append({
                't': global_t,
                'x': pos[j,0], 'y': pos[j,1], 'z': pos[j,2],
                'vx': vel[j,0], 'vy': vel[j,1], 'vz': vel[j,2],
                'ax': acc[j,0], 'ay': acc[j,1], 'az': acc[j,2],
                'type': segment_type
            })
            global_t += DT

# --- 5. SAVE & PLOT ---
df_final = pd.DataFrame(full_traj_rows)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Saved {len(df_final)} points.")

# VISUALIZATION
fig = plt.figure(figsize=(14, 8))

# Velocity Plot
ax1 = fig.add_subplot(211)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax1.plot(df_final['t'], speed, color='k', label='Speed')
ax1.fill_between(df_final['t'], 0, np.max(speed), where=df_final['type']==1, color='green', alpha=0.2, label='Draw')
ax1.set_title("Velocity Profile (Check Green Area)")
ax1.set_ylabel("Speed (m/s)")
ax1.legend()
ax1.grid(True)

# Path Plot (XY) - ดูว่ารายละเอียดกลับมาไหม
ax2 = fig.add_subplot(212)
ax2.scatter(df_final['x'], df_final['y'], c=df_final['type'], cmap='bwr', s=1)
ax2.set_title("XY Path (Blue=Draw, Red=Travel)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_aspect('equal')
ax2.grid(True)

plt.tight_layout()
plt.show()