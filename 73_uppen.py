import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_SafeLift.csv"

# Robot Settings
DT = 0.008 

# --- PHYSICS LIMITS (ปรับความนุ่มนวลตรงนี้) ---

# 1. ช่วงวาด (Draw)
LIMIT_VEL_DRAW = 0.05   # เดินช้าๆ ตอนวาด
LIMIT_ACC_DRAW = 0.1

# 2. ช่วงเดินทางราบ (XY Travel)
LIMIT_VEL_XY_TRAVEL = 0.25 # วิ่งเร็วได้ในแนวราบ
LIMIT_ACC_XY_TRAVEL = 0.5

# 3. *** ช่วงยก/วาง (Z Lift/Drop) *** -> แก้ตรงนี้ให้กราฟเอียง
LIMIT_VEL_Z_LIFT = 0.02    # (m/s) ยกช้าๆ (2cm ต่อวินาที) กราฟจะเอียงสวย
MIN_LIFT_TIME    = 0.5     # (s) บังคับว่าอย่างน้อยต้องใช้เวลา 0.5 วินาทีในการยก

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTIONS ---

def generate_quintic_travel(p_start, p_end, dt):
    """
    สร้าง Travel Path ที่แยกความเร็วระหว่าง XY (เร็ว) กับ Z (ช้า/นุ่ม)
    """
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    dist = np.linalg.norm(p_end - p_start)
    
    if dist < 1e-6: return None, None, None

    # แยกแยะประเภทการเคลื่อนที่
    xy_dist = np.linalg.norm(p_end[:2] - p_start[:2])
    z_dist = abs(p_end[2] - p_start[2])
    
    # Logic: ถ้า Z เปลี่ยนเยอะกว่า XY แสดงว่าเป็น Lift/Drop
    is_vertical_move = (z_dist > 0.001) and (xy_dist < 0.001)

    # --- คำนวณเวลา (Duration) ---
    if is_vertical_move:
        # ใช้ลิมิตความเร็วของแกน Z โดยเฉพาะ (ช้าๆ นุ่มๆ)
        # สูตร Quintic Time = 1.875 * Dist / MaxVel
        t_vel = (1.875 * dist) / LIMIT_VEL_Z_LIFT
        
        # เลือกเวลาที่นานที่สุด (ระหว่างคำนวณ กับ ขั้นต่ำ)
        duration = max(t_vel, MIN_LIFT_TIME)
        # print(f"-> Lifting Z... Time: {duration:.2f}s")
    else:
        # เดินทางแนวราบ ใช้ความเร็วสูงได้
        t_vel = (1.875 * dist) / LIMIT_VEL_XY_TRAVEL
        t_acc = np.sqrt((5.77 * dist) / LIMIT_ACC_XY_TRAVEL)
        duration = max(t_vel, t_acc, 0.1)

    # --- สร้าง Profile (Quintic Polynomial) ---
    # สูตรนี้การันตีว่า ความเร็วเริ่ม=0 และ จบ=0 (No Jerk at start/stop)
    num_steps = max(int(np.ceil(duration / dt)), 2)
    t = np.linspace(0, 1, num_steps)
    
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    dds = 60*t - 180*t**2 + 120*t**3
    
    pos = p_start + (p_end - p_start) * s[:, np.newaxis]
    vel = (p_end - p_start) * ds[:, np.newaxis] / duration
    acc = (p_end - p_start) * dds[:, np.newaxis] / (duration**2)
    
    return pos, vel, acc

def generate_linear_fallback(points, v_limit, dt):
    """Fallback: Linear Interpolation"""
    points = np.array(points)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_dist = np.sum(dists)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    duration = max(total_dist / v_limit, 0.1)
    t_points = (cum_dist / total_dist) * duration
    
    num_steps = max(int(np.ceil(duration / dt)), 2)
    t_eval = np.linspace(0, duration, num_steps)
    
    pos = np.zeros((num_steps, 3))
    vel = np.zeros((num_steps, 3))
    acc = np.zeros((num_steps, 3))
    
    for k in range(3):
        pos[:, k] = np.interp(t_eval, t_points, points[:, k])
        vel[:, k] = np.gradient(pos[:, k], dt)
        acc[:, k] = np.gradient(vel[:, k], dt)
    return pos, vel, acc

def generate_strict_continuous_draw(points, v_limit, a_limit, dt):
    """Draw: Spline"""
    points = np.array(points)
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid_mask = np.hstack(([True], diff > 1e-7)) 
    points = points[valid_mask]
    
    if len(points) < 3:
        if len(points) >= 2:
            return generate_quintic_travel(points[0], points[-1], dt)
        else:
            return None, None, None

    points[:, 2] = points[0, 2] # Flatten Z

    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_dist = np.sum(dists)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    current_duration = total_dist / v_limit * 1.2
    t_points_norm = cum_dist / total_dist
    
    success = False
    for _ in range(15):
        t_points_real = t_points_norm * current_duration
        try:
            cs = CubicSpline(t_points_real, points, axis=0, bc_type='clamped')
            num_steps = int(np.ceil(current_duration / dt))
            t_eval = np.linspace(0, current_duration, num_steps)
            acc_check = cs(t_eval, nu=2)
            a_peak = np.max(np.linalg.norm(acc_check, axis=1))
            
            if a_peak <= a_limit:
                success = True
                break
            else:
                ratio = np.sqrt(a_peak / a_limit)
                current_duration *= (ratio * 1.05)
        except:
            break

    if success:
        final_steps = max(int(np.ceil(current_duration / dt)), 2)
        t_final = np.linspace(0, current_duration, final_steps)
        return cs(t_final), cs(t_final, nu=1), cs(t_final, nu=2)
    else:
        return generate_linear_fallback(points, v_limit, dt)

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    p_next = waypoints[i+1]
    
    # 0 -> 1 : Start Drawing Chunk
    if (p_curr['type'] == 0 and p_next['type'] == 1):
        draw_chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        k = i + 1
        while k < len(waypoints):
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            if k+1 < len(waypoints) and waypoints[k+1]['type'] == 0:
                draw_chunk.append([waypoints[k+1]['x'], waypoints[k+1]['y'], waypoints[k+1]['z']])
                k += 1
                break
            k += 1
        
        pos, vel, acc = generate_strict_continuous_draw(draw_chunk, LIMIT_VEL_DRAW, LIMIT_ACC_DRAW, DT)
        i = k 
        segment_type = 1
        
    else:
        # 0 -> 0 : Travel / Lift / Drop
        start_pt = [p_curr['x'], p_curr['y'], p_curr['z']]
        end_pt   = [p_next['x'], p_next['y'], p_next['z']]
        
        # ใช้ฟังก์ชันใหม่ที่ไม่ต้องใส่ v_limit, a_limit (เพราะมันไปเลือกเองข้างใน)
        pos, vel, acc = generate_quintic_travel(start_pt, end_pt, DT)
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
fig = plt.figure(figsize=(14, 10))

# 1. Z-Profile (ZOOM IN)
# เราจะดูแค่ช่วงสั้นๆ เพื่อให้เห็นความชัน (Slope)
ax1 = fig.add_subplot(211)
subset = df_final.iloc[:2000] # ดูแค่ 2000 จุดแรก (ประมาณ 16 วินาทีแรก)
ax1.plot(subset['t'], subset['z'], color='blue', marker='.', markersize=2, label='Z Height')
ax1.set_title("Z-Height Profile (Zoomed In - Should see slope)")
ax1.set_ylabel("Z (m)")
ax1.grid(True)

# 2. Velocity Z Check
ax2 = fig.add_subplot(212)
ax2.plot(subset['t'], subset['vz'], color='orange', label='Z Velocity')
ax2.set_title("Z-Velocity (Check smoothness)")
ax2.set_ylabel("Vz (m/s)")
ax2.grid(True)

plt.tight_layout()
plt.show()