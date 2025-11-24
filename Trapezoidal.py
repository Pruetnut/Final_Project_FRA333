import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. ฟังก์ชันสร้าง Trajectory (Trapezoidal Generator)
# ---------------------------------------------------------
def trapezoidal_trajectory_generator(start_pos, end_pos, v_max, a_max, dt=0.01):
    """สร้างจุดย่อยๆ (x,y,z) ระหว่างจุดสองจุด ตาม profile ความเร็ว"""
    dist_vec = np.array(end_pos) - np.array(start_pos)
    distance = np.linalg.norm(dist_vec)
    
    if distance < 1e-6: # ถ้าระยะทางใกล้กันมาก (เช่นอยู่กับที่)
        return np.array([start_pos]), np.array([0])

    direction = dist_vec / distance 

    # คำนวณเวลา
    t_acc = v_max / a_max
    dist_acc = 0.5 * a_max * (t_acc**2)
    
    if distance < 2 * dist_acc: # ระยะสั้นเกินไป (Triangle Profile)
        t_acc = np.sqrt(distance / a_max)
        t_flat = 0
        v_peak = a_max * t_acc
    else: # ระยะปกติ (Trapezoid Profile)
        dist_flat = distance - 2 * dist_acc
        t_flat = dist_flat / v_max
        v_peak = v_max

    total_time = 2 * t_acc + t_flat
    time_steps = np.arange(0, total_time, dt)
    points = []
    
    for t in time_steps:
        s = 0
        if t <= t_acc:
            s = 0.5 * a_max * t**2
        elif t <= t_acc + t_flat:
            dist_pre_acc = 0.5 * a_max * t_acc**2
            s = dist_pre_acc + v_peak * (t - t_acc)
        else:
            t_remain = total_time - t
            s = distance - 0.5 * a_max * t_remain**2
            
        current_p = np.array(start_pos) + direction * s
        points.append(current_p)
        
    # แถมจุดสุดท้ายให้เป๊ะ
    points.append(end_pos)
    return np.array(points), total_time

# ---------------------------------------------------------
# 2. ฟังก์ชันจำลอง IK (Mock IK)
# ---------------------------------------------------------
def solve_ik_mock(x, y, z):
    # ในใช้งานจริงต้องเรียก ur_kinematics มาคำนวณ
    return [x*2, y*2, z*2, 0, -1.57, 0] # ค่าสมมติ

# ---------------------------------------------------------
# 3. Main Loop: รวมร่าง CSV + Trapezoidal + IK
# ---------------------------------------------------------

# โหลดไฟล์ CSV ที่ได้จากขั้นตอน Path Planning
df_waypoints = pd.read_csv('ur5_trajectory_commands_optimized.csv')

# กำหนดค่าความเร็ว/ความเร่ง (Tunable Parameters)
SPEED_CONFIG = {
    'Draw':  {'v': 0.02, 'a': 0.1},  # ช้าๆ นิ่งๆ (เมตร/วินาที)
    'Rapid': {'v': 0.20, 'a': 0.5},  # เร็วปานกลาง (เมตร/วินาที)
    'Slow':  {'v': 0.01, 'a': 0.1}   # สำหรับตอนลงปากกา
}

full_joint_trajectory = []
current_xyz = None # จะถูกเซ็ตค่าจากจุดแรกของไฟล์

# เวลาสะสม (Simulation Time)
sim_time = 0.0
DT = 0.008 # Time step ของ UR5 ปกติคือ 0.008s (125Hz) หรือ 0.002s (500Hz) แล้วแต่รุ่น

print(f"กำลังประมวลผล {len(df_waypoints)} waypoints...")

for index, row in df_waypoints.iterrows():
    # 1. ดึงเป้าหมาย (Target)
    target_xyz = np.array([row['X'], row['Y'], row['Z']])
    
    # กรณีรอบแรก (ยังไม่มีจุดเริ่มต้น)
    if current_xyz is None:
        current_xyz = target_xyz
        # ข้ามไปรอบถัดไปเลย เพราะยังเดินไม่ได้
        continue

    # 2. เลือกความเร็ว
    profile = row['Speed_Profile'] # 'Draw' หรือ 'Rapid'
    if profile not in SPEED_CONFIG: profile = 'Rapid' # Default
    
    params = SPEED_CONFIG[profile]
    
    # 3. สร้างจุดย่อยด้วย Trapezoidal (Interpolation)
    interp_points, duration = trapezoidal_trajectory_generator(
        current_xyz, target_xyz, 
        params['v'], params['a'], 
        dt=DT
    )
    
    # 4. วนลูปจุดย่อยเข้า IK
    for pt in interp_points:
        # เรียก IK (แปลง xyz เป็น joint angles)
        # หมายเหตุ: R, P, Y เราคงที่ไว้ตามไฟล์ CSV (row['Roll']...)
        joints = solve_ik_mock(pt[0], pt[1], pt[2])
        
        full_joint_trajectory.append({
            'Time': sim_time,
            'q1': joints[0], 'q2': joints[1], 'q3': joints[2],
            'q4': joints[3], 'q5': joints[4], 'q6': joints[5],
            'Action': row['Action'] # เก็บไว้ดูเล่น
        })
        
        sim_time += DT # เดินเวลา
        
    # 5. อัปเดตตำแหน่งปัจจุบัน
    current_xyz = target_xyz

# ---------------------------------------------------------
# 4. บันทึกผลลัพธ์
# ---------------------------------------------------------
df_final = pd.DataFrame(full_joint_trajectory)
df_final.to_csv('ur5_final_time_series.csv', index=False)

print(f"✅ เสร็จสิ้น! จาก {len(df_waypoints)} waypoints ขยายเป็น {len(df_final)} time steps")
print(f"เวลาที่ใช้ในการวาดทั้งหมด: {sim_time:.2f} วินาที")

# Plot ดูความเร็วแกน Z เพื่อเช็คจังหวะยกปากกา
plt.plot(df_final['Time'], df_final['q3'], label='Joint 3 (Elbow)')
plt.title("Joint Trajectory over Time")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.legend()
plt.show()