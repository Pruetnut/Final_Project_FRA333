import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_Logic_Fixed.csv"

# Robot Settings
DT = 0.008 

# Physics Limits
LIMIT_VEL_DRAW = 0.05   # วาดช้า (แม่น)
LIMIT_ACC_DRAW = 0.1

LIMIT_VEL_TRAVEL = 0.25 # ยกเร็ว
LIMIT_ACC_TRAVEL = 0.5

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
    """Travel: จุดต่อจุด (หยุดหัวท้าย)"""
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    dist = np.linalg.norm(p_end - p_start)
    
    # ถ้าจุดซ้ำ (เช่น Z_SAFE ลงมา Z_DRAW แล้ว x,y เท่าเดิมเป๊ะ)
    # ให้สร้างเวลาปลอมๆ นิดนึงเพื่อไม่ให้กราฟขาด
    if dist < 1e-6: 
        return np.array([p_start]), np.zeros((1,3)), np.zeros((1,3))

    # คำนวณเวลา (Minimum Jerk)
    t_vel = (1.875 * dist) / v_limit
    t_acc = np.sqrt((5.77 * dist) / a_limit)
    duration = max(t_vel, t_acc, 0.1) # ขั้นต่ำ 0.1s
    
    num_steps = max(int(np.ceil(duration / dt)), 2)
    t = np.linspace(0, 1, num_steps)
    
    # Quintic Equation
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    dds = 60*t - 180*t**2 + 120*t**3
    
    pos = p_start + (p_end - p_start) * s[:, np.newaxis]
    vel = (p_end - p_start) * ds[:, np.newaxis] / duration
    acc = (p_end - p_start) * dds[:, np.newaxis] / (duration**2)
    
    return pos, vel, acc

def generate_continuous_draw_chunk(points, v_limit, a_limit, dt):
    """Draw: เส้นต่อเนื่อง (Spline)"""
    points = np.array(points)
    
    # 1. CLEANING: ลบจุดซ้ำที่อยู่ติดกัน
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid_mask = np.hstack(([True], diff > 1e-6)) 
    points = points[valid_mask]
    
    # *** FALLBACK *** ถ้าจุดน้อยเกินไป ให้ใช้ Travel (เส้นตรง) แทน Spline
    # เพื่อป้องกันรายละเอียดหาย
    if len(points) < 3:
        if len(points) >= 2:
            return generate_quintic_travel(points[0], points[-1], v_limit, a_limit, dt)
        else:
            return None, None, None

    # 2. Iterative Time Scaling (แก้เวลาให้ Accel ไม่เกิน)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_dist = np.sum(dists)
    cum_dist = np.hstack(([0], np.cumsum(dists)))
    
    current_duration = total_dist / v_limit * 1.5 # เผื่อเวลาเริ่มต้น
    t_points_norm = cum_dist / total_dist
    
    success = False
    for _ in range(10): # ลองแก้ 10 รอบ
        t_points_real = t_points_norm * current_duration
        try:
            cs = CubicSpline(t_points_real, points, axis=0, bc_type='clamped')
            
            # Check Accel Limit
            num_steps = int(np.ceil(current_duration / dt))
            t_eval = np.linspace(0, current_duration, num_steps)
            acc_check = cs(t_eval, nu=2)
            a_peak = np.max(np.linalg.norm(acc_check, axis=1))
            
            if a_peak <= a_limit:
                success = True
                break
            else:
                # ยืดเวลา
                ratio = np.sqrt(a_peak / a_limit)
                current_duration *= (ratio * 1.05)
        except:
            break

    # ถ้า Spline พังจริงๆ ให้ใช้เส้นตรงเชื่อมหัวท้าย (Safety)
    if not success:
        return generate_quintic_travel(points[0], points[-1], v_limit, a_limit, dt)

    # Final Generate
    final_steps = max(int(np.ceil(current_duration / dt)), 2)
    t_final = np.linspace(0, current_duration, final_steps)
    
    pos = cs(t_final)
    vel = cs(t_final, nu=1)
    acc = cs(t_final, nu=2)
    
    return pos, vel, acc

# --- 4. MAIN LOOP (THE LOGIC YOU ASKED FOR) ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = waypoints[i]
    p_next = waypoints[i+1]
    
    # =========================================================
    # LOGIC: เช็คการเปลี่ยนถ่ายจาก 0 -> 1 (Start of Drawing)
    # =========================================================
    if (p_curr['type'] == 0 and p_next['type'] == 1):
        
        # 1. นี่คือจุดเริ่มวาด! รวบรวมจุดที่เป็นเส้นเดียวกัน (Chunking)
        # เริ่มเก็บที่จุด i (PenDown Point)
        draw_chunk = [[p_curr['x'], p_curr['y'], p_curr['z']]]
        
        k = i + 1
        while k < len(waypoints):
            # เก็บจุดถัดไป (ซึ่งควรเป็น Type 1)
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            
            # เช็คจุดต่อไปอีกที: ถ้าจุดต่อไปไม่ใช่ 1 (เช่นเป็น 0 คือจบเส้น) ให้หยุดเก็บ
            # หมายเหตุ: เราเก็บจุด Type 0 ตัวปิดท้ายไว้ด้วยก็ได้ เพื่อให้เส้นมันวิ่งไปจบที่ PenUp พอดี
            if k+1 < len(waypoints) and waypoints[k+1]['type'] == 0:
                # เก็บจุดปิดท้าย (Last Point of Line) แล้ว break
                draw_chunk.append([waypoints[k+1]['x'], waypoints[k+1]['y'], waypoints[k+1]['z']])
                k += 1 # ขยับ k ไปที่จุดปิดท้าย
                break
            
            k += 1
            
        # 2. ส่งทั้งก้อนไปคำนวณ Spline
        # print(f"Found Drawing Segment: {len(draw_chunk)} points")
        pos, vel, acc = generate_continuous_draw_chunk(draw_chunk, LIMIT_VEL_DRAW, LIMIT_ACC_DRAW, DT)
        
        # 3. ข้าม index ไปที่ปลายสุดของ Chunk
        i = k 
        segment_type = 1
        
    else:
        # =========================================================
        # LOGIC: Travel (0 -> 0)
        # =========================================================
        start_pt = [p_curr['x'], p_curr['y'], p_curr['z']]
        end_pt   = [p_next['x'], p_next['y'], p_next['z']]
        
        pos, vel, acc = generate_quintic_travel(start_pt, end_pt, LIMIT_VEL_TRAVEL, LIMIT_ACC_TRAVEL, DT)
        
        i += 1
        segment_type = 0

    # บันทึกข้อมูล
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
fig = plt.figure(figsize=(12, 10))

# 1. Velocity Profile
ax1 = fig.add_subplot(211)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax1.plot(df_final['t'], speed, color='k', label='Speed')
ax1.fill_between(df_final['t'], 0, np.max(speed), where=df_final['type']==1, color='green', alpha=0.3, label='Drawing')
ax1.set_title("Velocity Profile (Separated by Logic 0->1)")
ax1.set_ylabel("Speed (m/s)")
ax1.legend()
ax1.grid(True)

# 2. Path (XY View) เพื่อดูว่าเส้นหายไหม
ax2 = fig.add_subplot(212)
ax2.scatter(df_final['x'], df_final['y'], c=df_final['type'], cmap='bwr', s=1)
ax2.set_title("XY Path (Blue=Draw, Red=Travel)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_aspect('equal')
ax2.grid(True)

plt.tight_layout()
plt.show()