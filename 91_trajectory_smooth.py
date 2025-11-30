import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Matlab/fWaypoints_fix_doubleline_hand_edit.csv" 
OUTPUT_CSV = "Matlab/f_Trajectory.csv"

# Robot Settings (UR5e Specs)
DT = 0.008  # Sampling time 8ms (125Hz)

# Operational Limits (ตั้งค่าใช้งานจริง ไม่ใช่ Max Spec)
# UR5e Max Joint Speed ~3.14 rad/s, Max TCP Speed ~1 m/s (Safety)
# แต่สำหรับงานวาด เราต้องจำกัดให้ช้าลงเพื่อความแม่นยำ

LIMIT_VEL_DRAW = 0.05    # (5 cm/s)
LIMIT_ACC_DRAW = 0.1     # (10 cm/s^2)
LIMIT_VEL_TRAVEL = 0.25  # (25 cm/s)
LIMIT_ACC_TRAVEL = 0.5   # (50 cm/s^2)
# Force Time for Vertical Moves
MIN_LIFT_DURATION = 0.9  # (0.9s)

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found. Please run the waypoint generation script first.")
    exit()

df = pd.read_csv(INPUT_CSV)
# Ensure type is int
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTIONS ---

def generate_quintic_travel(p_start, p_end, v_limit, a_limit, dt):
    """
    สร้าง Travel Path แบบ Point-to-Point (หยุดหัวท้าย)
    ใช้ Quintic Polynomial: s(t) = 10t^3 - 15t^4 + 6t^5
    """
    p_start = np.array([p_start['x'], p_start['y'], p_start['z']])
    p_end = np.array([p_end['x'], p_end['y'], p_end['z']])
    
    dist = np.linalg.norm(p_end - p_start)
    
    # ถ้าจุดซ้ำกัน (Distance ~ 0) ให้ข้าม
    if dist < 1e-6: 
        return None, None, None

    # Logic: ตรวจสอบว่าเป็น Vertical Move หรือไม่ (Z เปลี่ยน, XY แทบไม่เปลี่ยน)
    xy_dist = np.linalg.norm(p_end[:2] - p_start[:2])
    z_dist = abs(p_end[2] - p_start[2])
    is_vertical = (z_dist > 0.001) and (xy_dist < 0.001)

    # 1. คำนวณเวลา (Duration)
    if is_vertical:
        # ถ้าเป็นการยก/วาง ให้ใช้เวลาตาม Speed Limit หรือเวลาขั้นต่ำ (Force Lift Time)
        t_calc = (1.875 * dist) / 0.05 # ใช้ความเร็วต่ำมากสำหรับแกน Z โดยเฉพาะ (เพื่อความชัวร์)
        duration = max(t_calc, MIN_LIFT_DURATION)
    else:
        # ถ้าเป็นการเดินทางปกติ
        t_vel = (1.875 * dist) / v_limit
        t_acc = np.sqrt((5.77 * dist) / a_limit)
        duration = max(t_vel, t_acc, 0.1) # ขั้นต่ำ 0.1s

    # 2. สร้าง Time Steps
    num_steps = max(int(np.ceil(duration / dt)), 2)
    t = np.linspace(0, 1, num_steps) # Normalized time 0->1
    
    # 3. Quintic Equation
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    dds = 60*t - 180*t**2 + 120*t**3
    
    # 4. Calculate Kinematics
    pos = p_start + (p_end - p_start) * s[:, np.newaxis]
    vel = (p_end - p_start) * ds[:, np.newaxis] / duration
    acc = (p_end - p_start) * dds[:, np.newaxis] / (duration**2)
    
    return pos, vel, acc

def generate_cubic_spline_draw(points, v_limit, a_limit, dt):
    """
    สร้าง Drawing Path แบบต่อเนื่อง (Continuous)
    ใช้ Cubic Spline with Clamped BC (หยุดแค่หัวกับท้ายเส้น)
    """
    if len(points) < 3: return None, None, None
    
    # แปลงเป็น Numpy Array
    coords = np.array([[p['x'], p['y'], p['z']] for p in points])
    
    # 1. Cleaning: ลบจุดซ้ำที่อยู่ติดกัน
    diff = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    valid_mask = np.hstack(([True], diff > 1e-6))
    coords = coords[valid_mask]
    
    if len(coords) < 2: return None, None, None # ถ้าเหลือจุดเดียว ทำ Spline ไม่ได้

    # 2. Flatten Z (บังคับ Z ให้เท่ากันตลอดเส้นวาด เพื่อป้องกันหัวปากกากระดก)
    coords[:, 2] = coords[0, 2]

    # 3. Time Parameterization (Estimate Time based on Distance)
    dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    total_dist = np.sum(dists)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    # เวลาเริ่มต้น (Safety Factor 1.5x)
    current_duration = total_dist / v_limit * 1.5
    
    # Normalized time (0->1) ตามระยะทาง
    t_points_norm = cum_dist / total_dist
    
    # 4. Iterative Time Scaling (วนลูปแก้เวลาจนกว่า V, A จะไม่เกิน Limit)
    success = False
    for _ in range(15):
        t_points_real = t_points_norm * current_duration
        try:
            # Clamped BC: บังคับความเร็วต้นและปลายเส้นเป็น 0
            cs = CubicSpline(t_points_real, coords, axis=0, bc_type='clamped')
            
            # Check Physics Constraints
            num_steps = int(np.ceil(current_duration / dt))
            if num_steps < 2: num_steps = 2
            t_eval = np.linspace(0, current_duration, num_steps)
            
            vel_check = cs(t_eval, nu=1)
            acc_check = cs(t_eval, nu=2)
            
            v_peak = np.max(np.linalg.norm(vel_check, axis=1))
            a_peak = np.max(np.linalg.norm(acc_check, axis=1))
            
            # ถ้าผ่านเกณฑ์ (หรือเกินนิดหน่อย < 1%) ให้จบ
            if v_peak <= v_limit * 1.01 and a_peak <= a_limit * 1.01:
                success = True
                break
            else:
                # ถ้าไม่ผ่าน ให้ยืดเวลาเพิ่ม (Scaling Time)
                # a ~ 1/t^2, v ~ 1/t -> เลือกตัวที่แย่ที่สุดมาคำนวณ Ratio
                ratio_v = v_peak / v_limit
                ratio_a = np.sqrt(a_peak / a_limit)
                ratio = max(ratio_v, ratio_a)
                current_duration *= (ratio * 1.05) # เพิ่ม Buffer 5%
        except:
            break

    # 5. Generate Final Trajectory
    final_steps = max(int(np.ceil(current_duration / dt)), 2)
    t_final = np.linspace(0, current_duration, final_steps)
    
    if success:
        pos = cs(t_final)
        vel = cs(t_final, nu=1)
        acc = cs(t_final, nu=2)
        return pos, vel, acc
    else:
        # Fallback: ถ้า Spline พังจริงๆ ให้ใช้ Linear Interpolation
        # (เก็บ Shape ไว้ แต่ความเร็วอาจจะไม่นิ่งเท่า)
        pos = np.zeros((final_steps, 3))
        for k in range(3):
            pos[:, k] = np.interp(t_final, t_points_norm * current_duration, coords[:, k])
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        return pos, vel, acc

# --- 4. MAIN LOOP ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    p_next = waypoints[i+1]
    
    # LOGIC 1: เริ่มเข้าสู่โหมดวาด (Draw Chunk)
    # เงื่อนไข: จุดปัจจุบันเป็น Type 0 (Pen Down) -> จุดต่อไปเป็น Type 1 (เริ่มวาด)
    if (p_curr['type'] == 0 and p_next['type'] == 1):
        
        # รวบรวมจุดวาดทั้งหมดมาเป็นก้อนเดียว (Chunking)
        draw_chunk = [p_curr] # เริ่มที่จุด Pen Down
        k = i + 1
        while k < len(waypoints):
            draw_chunk.append(waypoints[k])
            # ถ้าเจอจุดที่ไม่ใช่ Type 1 (เช่นจบเส้นแล้วเตรียมยก) ให้หยุดเก็บ
            # แต่เก็บจุด Type 0 ปิดท้ายไว้ด้วยเพื่อให้เส้นสมบูรณ์
            if waypoints[k]['type'] == 0: 
                break 
            k += 1
            
        # สร้าง Continuous Spline
        pos, vel, acc = generate_cubic_spline_draw(draw_chunk, LIMIT_VEL_DRAW, LIMIT_ACC_DRAW, DT)
        
        # อัปเดต Index ข้ามไปที่ปลาย Chunk
        i = k 
        segment_type = 1
        
    # LOGIC 2: โหมดเดินทาง (Travel / Lift / Lower)
    else:
        # สร้าง Point-to-Point Quintic Move
        pos, vel, acc = generate_quintic_travel(p_curr, p_next, LIMIT_VEL_TRAVEL, LIMIT_ACC_TRAVEL, DT)
        
        i += 1
        segment_type = 0

    # บันทึกข้อมูลลง List
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

# --- 5. FINAL OUTPUT & CHECK ---
df_final = pd.DataFrame(full_traj_rows)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Generated Trajectory: {len(df_final)} points.")
print(f"   Total Duration: {global_t:.2f} seconds")

# --- VISUALIZATION ---
fig = plt.figure(figsize=(14, 10))

# 1. Velocity Profile (Check Smoothness)
ax1 = fig.add_subplot(211)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax1.plot(df_final['t'], speed, 'k-', linewidth=1, label='Speed')
# Highlight Draw areas
ax1.fill_between(df_final['t'], 0, speed.max(), where=df_final['type']==1, color='green', alpha=0.2, label='Drawing Phase')
ax1.set_title("Velocity Profile (Green=Drawing, White=Travel)")
ax1.set_ylabel("Speed (m/s)")
ax1.legend()
ax1.grid(True)

# 2. Z-Height Profile (Check Lifts)
ax2 = fig.add_subplot(212)
ax2.plot(df_final['t'], df_final['z'], 'b-', linewidth=1.5)
ax2.set_title("Z-Height Profile (Check Vertical Lifts)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Z (m)")
ax2.grid(True)

plt.tight_layout()
plt.show()