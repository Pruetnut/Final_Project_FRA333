import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_FixedDraw.csv"

# Robot Settings
DT = 0.008 

# Physics Limits
LIMIT_VEL_DRAW = 0.02
LIMIT_ACC_DRAW = 0.5

LIMIT_VEL_TRAVEL = 0.1
LIMIT_ACC_TRAVEL = 0.1

MIN_LIFT_DURATION = 100 # บังคับเวลายก

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTIONS ---

def generate_quintic_travel(p_start, p_end, v_limit, a_limit, dt):
    """Travel: จุดต่อจุด (หยุดหัวท้าย) สำหรับช่วงยก/วาง"""
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    dist = np.linalg.norm(p_end - p_start)
    
    if dist < 1e-6: return None, None, None

    # --- แก้ไขตรงนี้ ---
    # เช็คว่ามีการขยับในแกน Z หรือไม่? (ถ้า Z เปลี่ยน แปลว่ากำลังยกหรือวาง)
    z_diff = abs(p_end[2] - p_start[2])
    
    # ถ้า Z เปลี่ยนมากกว่า 1 มม. ให้ถือว่าเป็น Vertical Move (Lift/Drop)
    # ไม่ต้องสน XY มากนัก (เผื่อ XY ขยับนิดหน่อยตอนยก)
    is_lift_or_drop = z_diff > 0.001 

    if is_lift_or_drop:
        # ถ้าเป็นการยก/วาง ให้ใช้เวลาอย่างน้อย MIN_LIFT_DURATION (เช่น 0.5s หรือ 10s ตามที่คุณตั้ง)
        # แต่ถ้าคำนวณตามความเร็วมันนานกว่า ก็ให้ใช้ค่าที่นานกว่า
        duration_calc = (1.875 * dist) / v_limit
        duration = max(duration_calc, MIN_LIFT_DURATION) 
    else:
        # เดินทางราบ (Travel XY)
        t_vel = (1.875 * dist) / v_limit
        t_acc = np.sqrt((5.77 * dist) / a_limit)
        duration = max(t_vel, t_acc, 0.1)

    # (ส่วนที่เหลือเหมือนเดิม)
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

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    p_next = waypoints[i+1]
    
    # LOGIC 0 -> 1: เริ่มวาด
    if (p_curr['type'] == 0 and p_next['type'] == 1):
        # Draw Chunk
        draw_chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        k = i + 1
        while k < len(waypoints):
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            if k+1 < len(waypoints) and waypoints[k+1]['type'] == 0:
                # ปิดท้ายด้วยจุดจบเส้น (เพื่อให้เส้นคม)
                draw_chunk.append([waypoints[k+1]['x'], waypoints[k+1]['y'], waypoints[k+1]['z']])
                k += 1
                break
            k += 1
            
        pos, vel, acc = generate_strict_continuous_draw(draw_chunk, LIMIT_VEL_DRAW, LIMIT_ACC_DRAW, DT)
        i = k 
        segment_type = 1
        
    else:
        # Travel / Lift / Drop
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
fig = plt.figure(figsize=(14, 10))

# 1. 3D Path Check (สำคัญ: ดูว่ารูปวาดมาครบไหม)
ax1 = fig.add_subplot(211, projection='3d')
# Plot Travel (Red)
travel_mask = df_final['type'] == 0
ax1.scatter(df_final.loc[travel_mask, 'x'], df_final.loc[travel_mask, 'y'], df_final.loc[travel_mask, 'z'], 
            c='red', s=1, label='Travel')
# Plot Draw (Green)
draw_mask = df_final['type'] == 1
ax1.scatter(df_final.loc[draw_mask, 'x'], df_final.loc[draw_mask, 'y'], df_final.loc[draw_mask, 'z'], 
            c='green', s=1, label='Draw')

ax1.set_title("3D Trajectory (Green lines must form the image)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_box_aspect([1,1,0.5])
ax1.legend()

# 2. Z-Profile (Check Lift)
ax2 = fig.add_subplot(212)
ax2.plot(df_final['t'], df_final['z'], color='blue')
ax2.set_title("Z-Height Profile (Check Lifts)")
ax2.set_ylabel("Z (m)")
ax2.grid(True)

plt.tight_layout()
plt.show()