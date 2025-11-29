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


# Load Data
df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
# ต้องมี column 'cmd' มาด้วย
waypoints = df.to_dict('records')

# --- Helper Functions (ปรับปรุง) ---
def generate_linear_fallback(points, v_limit, dt):
    """
    *** แผนสำรองใหม่ ***
    ถ้า Spline พัง ให้ใช้ Linear (ลากจุดต่อจุด) แทนการลากเส้นตรงหัวท้าย
    รับประกันว่ารูปร่าง (Shape) จะยังอยู่ครบ
    """
    points = np.array(points)
    
    # คำนวณระยะทางสะสม
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    total_dist = cum_dist[-1]
    
    # คำนวณเวลา (ใช้ความเร็วคงที่)
    duration = max(total_dist / v_limit, 0.1)
    
    # สร้าง Time vector สำหรับแต่ละจุด
    t_points = (cum_dist / total_dist) * duration
    
    # Resample
    num_steps = max(int(np.ceil(duration / dt)), 2)
    t_eval = np.linspace(0, duration, num_steps)
    
    pos = np.zeros((num_steps, 3))
    vel = np.zeros((num_steps, 3))
    acc = np.zeros((num_steps, 3)) # Linear ความเร่งเป็น 0 (ยกเว้นตรงมุม)
    
    for k in range(3): # x, y, z
        pos[:, k] = np.interp(t_eval, t_points, points[:, k])
        vel[:, k] = np.gradient(pos[:, k], dt)
        acc[:, k] = np.gradient(vel[:, k], dt)
        
    return pos, vel, acc

def generate_strict_continuous_draw(points, v_limit, a_limit, dt):
    """Draw: Spline ต่อเนื่อง (พยายามทำให้สมูทที่สุด)"""
    points = np.array(points)
    
    # 1. Cleaning (ผ่อนปรนลง)
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # กรองเฉพาะจุดที่ซ้ำกันจริงๆ (ระยะเกือบ 0)
    valid_mask = np.hstack(([True], diff > 1e-7)) 
    points = points[valid_mask]
    
    # Fallback ถ้าจุดน้อย
    if len(points) < 3:
        if len(points) >= 2:
            return generate_quintic_travel(points[0], points[-1], v_limit, a_limit, dt)
        else:
            return None, None, None

    # Flatten Z
    points[:, 2] = points[0, 2]

    # 2. Try Cubic Spline
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_dist = np.sum(dists)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    current_duration = total_dist / v_limit * 1.2
    t_points_norm = cum_dist / total_dist
    
    success = False
    
    # พยายามแก้เวลา (Time Scaling)
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
        # ถ้า Spline ผ่าน ใช้ Spline
        final_steps = max(int(np.ceil(current_duration / dt)), 2)
        t_final = np.linspace(0, current_duration, final_steps)
        pos = cs(t_final)
        vel = cs(t_final, nu=1)
        acc = cs(t_final, nu=2)
        return pos, vel, acc
    else:
        # *** ถ้า Spline ไม่ผ่าน ให้ใช้ Linear Fallback (เก็บทรงไว้) ***
        # print("Spline failed, using Linear Fallback (Shape Preserved)")
        return generate_linear_fallback(points, v_limit, dt)

def generate_quintic_move(p_start, p_end, v_limit, a_limit, dt, forced_time=None):
    """
    สร้าง Quintic Path โดยรับเวลาแบบ Force ได้
    """
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    dist = np.linalg.norm(p_end - p_start)
    
    # ถ้ามีเวลาบังคับมา ให้ใช้เลย (สำหรับ Lift/Lower)
    if forced_time is not None:
        duration = forced_time
    else:
        # คำนวณปกติ
        if dist < 1e-6: return None, None, None
        t_vel = (1.875 * dist) / v_limit
        t_acc = np.sqrt((5.77 * dist) / a_limit)
        duration = max(t_vel, t_acc, 0.1)

    num_steps = max(int(np.ceil(duration / dt)), 2)
    t = np.linspace(0, 1, num_steps)
    
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    dds = 60*t - 180*t**2 + 120*t**3
    
    pos = p_start + (p_end - p_start) * s[:, np.newaxis]
    vel = (p_end - p_start) * ds[:, np.newaxis] / duration
    acc = (p_end - p_start) * dds[:, np.newaxis] / (duration**2)
    
    return pos, vel, acc

# ... (generate_strict_continuous_draw เหมือนเดิม) ...

# --- MAIN LOOP (Logic ใหม่ตาม Command) ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    p_next = waypoints[i+1]
    
    # อ่าน Command ของจุดถัดไป (เรากำลังจะทำอะไร?)
    next_cmd = p_next.get('cmd', 'UNKNOWN')
    
    if p_curr['type'] == 0 and p_next['type'] == 1:
        # === DRAW CHUNK === (เหมือนเดิม)
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
        # === DISCRETE MOVE (Type 0 -> 0) ===
        start_pt = [p_curr['x'], p_curr['y'], p_curr['z']]
        end_pt   = [p_next['x'], p_next['y'], p_next['z']]
        
        # เลือกเวลาตาม Command
        force_t = None
        if next_cmd in ['LIFT', 'LOWER']:
            force_t = 0.5  # บังคับเวลาตรงนี้! (เช่น 0.5s)
            
        pos, vel, acc = generate_quintic_move(
            start_pt, end_pt, 
            0.05,1,         #LIMIT_VEL_TRAVEL, LIMIT_ACC_TRAVEL, 
            DT, 
            forced_time=force_t # ส่งเวลาบังคับเข้าไป
        )
        
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