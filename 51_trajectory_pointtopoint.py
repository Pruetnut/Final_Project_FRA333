import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ruckig import Ruckig, InputParameter, OutputParameter, Result
import time
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"       # ไฟล์ Input จากขั้นตอนที่แล้ว
OUTPUT_CSV = "Final_Trajectory_Full_Data.csv" # ไฟล์ Output ที่สมบูรณ์

# Robot & Ruckig Settings
CONTROL_CYCLE = 0.02  # 8ms (มาตรฐาน UR5)

# Limits (ปรับความเร็ว/ความนุ่มนวลตรงนี้)
# หน่วย: m/s, m/s^2, m/s^3
MAX_VEL = [0.5, 0.5, 0.5]    
MAX_ACC = [0.8, 0.8, 0.8]      
MAX_JERK = [5.0, 5.0, 5.0]     # ยิ่งน้อยยิ่งนุ่ม (แต่เข้าโค้งช้า), ยิ่งมากยิ่งตอบสนองไว

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: ไม่พบไฟล์ {INPUT_CSV} กรุณารันไฟล์ก่อนหน้าเพื่อสร้าง Map ก่อน")
    exit()

# อ่านไฟล์ Waypoints
df_waypoints = pd.read_csv(INPUT_CSV)
waypoints = df_waypoints[['x', 'y', 'z', 'type']].to_dict('records')

print(f"Loaded {len(waypoints)} waypoints from {INPUT_CSV}")

# --- 3. INITIALIZE RUCKIG ---
otg = Ruckig(3, CONTROL_CYCLE) # 3 DOFs (X, Y, Z)
inp = InputParameter(3)
out = OutputParameter(3)

# Set Kinematic Limits
inp.max_velocity = MAX_VEL
inp.max_acceleration = MAX_ACC
inp.max_jerk = MAX_JERK

# Set Initial State (เริ่มที่จุดแรกของ Waypoint)
first_wp = waypoints[0]
inp.current_position = [first_wp['x'], first_wp['y'], first_wp['z']]
inp.current_velocity = [0.0, 0.0, 0.0]
inp.current_acceleration = [0.0, 0.0, 0.0]

# ตัวแปรสำหรับเก็บผลลัพธ์
full_traj_data = []
global_time = 0.0

print("Start generating trajectory... (This might take a moment)")
start_process_time = time.time()

# --- 4. MAIN GENERATION LOOP ---
for i in range(1, len(waypoints)):
    target = waypoints[i]
    
    # กำหนดเป้าหมายใหม่ (Target)
    inp.target_position = [target['x'], target['y'], target['z']]
    
    # Tip: สำหรับงานวาดที่จุดถี่ๆ เราตั้ง Target Vel เป็น 0 (Stop-and-Go แบบ Micro)
    # แต่ด้วย Jerk ที่จำกัด มันจะไม่หยุดกึกกัก แต่จะ flow ไปเอง
    inp.target_velocity = [0.0, 0.0, 0.0] 
    inp.target_acceleration = [0.0, 0.0, 0.0]
    
    # Ruckig Loop (คำนวณไส้ในระหว่างจุด A -> B)
    while True:
        # Update Ruckig
        result = otg.update(inp, out)
        
        # บันทึกข้อมูลลง List
        full_traj_data.append({
            't': global_time,
            'x': out.new_position[0], 'y': out.new_position[1], 'z': out.new_position[2],
            'vx': out.new_velocity[0], 'vy': out.new_velocity[1], 'vz': out.new_velocity[2],
            'ax': out.new_acceleration[0], 'ay': out.new_acceleration[1], 'az': out.new_acceleration[2],
            'type': target['type'] # แปะป้ายไว้ว่าช่วงนี้กำลังทำอะไร
        })
        
        # ส่งค่า Output กลับไปเป็น Input ของรอบถัดไป
        out.pass_to_input(inp)
        global_time += CONTROL_CYCLE
        
        # เช็คว่าถึงเป้าหมายหรือยัง
        if result == Result.Finished:
            break
            
    # (Optional) Print progress ทุกๆ 100 จุด
    if i % 100 == 0:
        print(f"Processed waypoints: {i}/{len(waypoints)}")

end_process_time = time.time()
print(f"Generation Complete in {end_process_time - start_process_time:.2f} seconds!")

# --- 5. SAVE & VISUALIZE ---
df_final = pd.DataFrame(full_traj_data)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Saved full trajectory to: {OUTPUT_CSV} ({len(df_final)} steps)")

# === Plotting ============================================================================
# 1. 3D Path
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')

# Downsample ข้อมูลตอนพลอตเพื่อความเร็ว (เอาทุกๆ 10 จุด)
plot_df = df_final.iloc[::10]

# แยกสี: แดง=ยก(Travel), น้ำเงิน=วาด(Draw)
colors = ['red' if t in ['TRAVEL', 'PEN_UP', 'HOME'] else 'blue' for t in plot_df['type']]

ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1)
ax1.set_title("Generated Ruckig Trajectory")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# Fix Aspect Ratio
max_range = np.array([plot_df.x.max()-plot_df.x.min(), plot_df.y.max()-plot_df.y.min()]).max() / 2.0
mid_x = (plot_df.x.max()+plot_df.x.min()) * 0.5
mid_y = (plot_df.y.max()+plot_df.y.min()) * 0.5
ax1.set_xlim(mid_x - max_range, mid_x + max_range)
ax1.set_ylim(mid_y - max_range, mid_y + max_range)

# 2. Velocity Profile (Z-axis) เพื่อดูการ ยก/วาง
ax2 = fig.add_subplot(122)
ax2.plot(df_final['t'], df_final['vz'], label='Vz (Vertical Vel)', color='green')
ax2.plot(df_final['t'], df_final['vx'], label='Vx', alpha=0.3)
ax2.set_title("Velocity Profile (Check Smoothness)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()