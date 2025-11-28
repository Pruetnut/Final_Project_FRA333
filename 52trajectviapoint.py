import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ruckig import Ruckig, InputParameter, OutputParameter, Result
import time
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_ViaPoints.csv"

# Robot Settings
CONTROL_CYCLE = 0.008

# Limits (ปรับได้)
MAX_VEL = [0.25, 0.25, 0.25]
MAX_ACC = [1.0, 1.0, 1.0]      # เพิ่ม Acc ได้เพื่อให้เกาะเส้นดีขึ้นตอนเข้าโค้ง
MAX_JERK = [5.0, 5.0, 5.0]     # เพิ่ม Jerk ได้เพื่อให้เปลี่ยนทิศทางไวขึ้น

# *** หัวใจสำคัญของ Via Point ***
# ถ้าระยะห่างถึงเป้าหมาย น้อยกว่าค่านี้ ให้ถือว่า "ผ่านแล้ว" และไปจุดต่อไปทันที
BLENDING_RADIUS = 0.002  # 2 มิลลิเมตร (ถ้าอยากให้โค้งมนมาก ให้เพิ่มค่านี้)

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df_waypoints = pd.read_csv(INPUT_CSV)
waypoints = df_waypoints.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. INITIALIZE RUCKIG ---
otg = Ruckig(3, CONTROL_CYCLE)
inp = InputParameter(3)
out = OutputParameter(3)

inp.max_velocity = MAX_VEL
inp.max_acceleration = MAX_ACC
inp.max_jerk = MAX_JERK

# Set Initial State
first_wp = waypoints[0]
inp.current_position = [first_wp['x'], first_wp['y'], first_wp['z']]
inp.current_velocity = [0.0, 0.0, 0.0]
inp.current_acceleration = [0.0, 0.0, 0.0]

full_traj_data = []
global_time = 0.0

print("Generating Continuous Trajectory...")

# --- 4. CONTINUOUS TRAJECTORY LOOP ---
# เราจะวนลูปทีละจุด แต่จะเช็คเงื่อนไขต่างกัน
i = 1
while i < len(waypoints):
    target = waypoints[i]
    next_target = waypoints[i+1] if i+1 < len(waypoints) else None
    
    # กำหนดเป้าหมาย
    inp.target_position = [target['x'], target['y'], target['z']]
    inp.target_velocity = [0.0, 0.0, 0.0]
    inp.target_acceleration = [0.0, 0.0, 0.0]
    
    # --- CHECKPOINT LOGIC: ตัดสินใจว่าเป็น Via Point หรือ Stop Point ---
    is_via_point = False
    
    # ถ้าจุดปัจจุบันเป็น DRAW และจุดต่อไปก็เป็น DRAW -> ให้เป็น Via Point (บินผ่าน)
    if (target['type'] == 'DRAW') and (next_target and next_target['type'] == 'DRAW'):
        is_via_point = True
        
    # หมายเหตุ: จุด PEN_DOWN, PEN_UP, TRAVEL หรือจุดสุดท้ายของเส้น DRAW จะเป็น Stop Point เสมอ
    
    # Ruckig Loop ภายใน Segment นี้
    while True:
        result = otg.update(inp, out)
        
        # บันทึกข้อมูล
        full_traj_data.append({
            't': global_time,
            'x': out.new_position[0], 'y': out.new_position[1], 'z': out.new_position[2],
            'vx': out.new_velocity[0], 'vy': out.new_velocity[1], 'vz': out.new_velocity[2],
            'type': target['type']
        })
        
        # ส่งค่ากลับไปเป็น Input รอบหน้า (รักษาโมเมนตัมความเร็วไว้!)
        out.pass_to_input(inp)
        global_time += CONTROL_CYCLE
        
        # --- เงื่อนไขการเปลี่ยนจุด (Switching Logic) ---
        
        current_pos = np.array(out.new_position)
        target_pos = np.array(inp.target_position)
        dist_to_target = np.linalg.norm(target_pos - current_pos)
        
        if is_via_point:
            # CASE 1: VIA POINT
            # ถ้าเข้าใกล้รัศมี Blending Radius -> เปลี่ยนไปจุดต่อไปเลย (ไม่ต้องรอหยุด)
            if dist_to_target < BLENDING_RADIUS:
                break  # Break เพื่อไปรับ i ถัดไป
        else:
            # CASE 2: STOP POINT
            # ต้องรอจนกว่า Ruckig จะบอกว่าจบ (ความเร็วเป็น 0 สนิท)
            if result == Result.Finished:
                break
                
    i += 1 # ขยับไปจุดถัดไป

# --- 5. SAVE & PLOT ---
df_final = pd.DataFrame(full_traj_data)
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Saved {len(df_final)} points.")

# Visualization
fig = plt.figure(figsize=(12, 6))

# Plot 1: 3D Path
ax1 = fig.add_subplot(121, projection='3d')
plot_df = df_final.iloc[::5] # Downsample
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=plot_df['vz'], cmap='coolwarm', s=1)
ax1.set_title("Continuous Path (Via Points)")

# Plot 2: Velocity Profile (เทียบกัน)
ax2 = fig.add_subplot(122)
# ความเร็วรวม (Linear Velocity)
v_norm = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], v_norm, label='Speed (m/s)', color='purple')
ax2.set_title("Velocity Profile (Should NOT drop to zero during drawing)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (m/s)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()