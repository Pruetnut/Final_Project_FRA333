import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_With_Stops.csv"

# Robot Settings
DT = 0.008  # Sampling time

# Limits
SPEED_TRAVEL = 0.25 # m/s (วิ่งเร็วตอนยกปากกา)
SPEED_DRAW   = 0.05 # m/s (วาดช้าๆ)
ACCEL_TIME   = 0.05 # เวลาเร่งความเร็ว

# --- 2. LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)
waypoints = df.to_dict('records')
print(f"Loaded {len(waypoints)} waypoints.")

# --- 3. HELPER FUNCTIONS ---

def generate_travel_segment(p_start, p_end, dt, max_speed):
    """สร้างเส้นทางแบบจุดต่อจุด (เริ่ม 0 -> เร่ง -> จบ 0)"""
    dist = np.linalg.norm(p_end - p_start)
    if dist < 1e-6: return None, None, None
    
    # คำนวณเวลาที่ต้องใช้
    duration = max(dist / max_speed, 0.5) # อย่างน้อย 0.5 วิ เพื่อความนิ่ม
    t_steps = np.arange(0, duration, dt)
    
    # ใช้ jtraj (Minimum Jerk)
    traj = rtb.jtraj(p_start, p_end, t_steps)
    return traj.q, traj.qd, traj.qdd

def generate_drawing_segment(points, dt, max_speed):
    """สร้างเส้นทางวาดต่อเนื่อง (เริ่ม 0 -> ไหล... -> จบ 0)"""
    if len(points) < 2: return None, None, None
    
    # ใช้ mstraj สำหรับ Multi-segment
    traj = rtb.mstraj(np.array(points), dt=dt, tacc=ACCEL_TIME, qdmax=max_speed)
    
    pos = traj.q
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return pos, vel, acc

# --- 4. MAIN GENERATION LOOP (HYBRID) ---
full_traj_rows = []
global_t = 0.0

i = 0
while i < len(waypoints) - 1:
    p_curr = np.array([waypoints[i]['x'], waypoints[i]['y'], waypoints[i]['z']])
    
    # LOGIC: เช็คว่าเรากำลังจะเข้าโหมด "วาดต่อเนื่อง" (Draw Chunk) หรือไม่
    # เงื่อนไข: จุดปัจจุบันเป็น Type 1 และจุดถัดไปก็เป็น Type 1
    if waypoints[i]['type'] == 1 and waypoints[i+1]['type'] == 1:
        
        # --- MODE 1: DRAWING (ต่อเนื่องในเส้น แต่หยุดหัวท้าย) ---
        draw_chunk = [p_curr]
        k = i + 1
        while k < len(waypoints) and waypoints[k]['type'] == 1:
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            k += 1
            
        # สร้าง Trajectory วาด (Chunk นี้จะเริ่มที่ 0 และจบที่ 0 เองโดยธรรมชาติของ mstraj)
        pos, vel, acc = generate_drawing_segment(draw_chunk, DT, SPEED_DRAW)
        
        # ข้าม index ไป
        i = k - 1
        segment_type = 1
        
    else:
        # --- MODE 2: TRAVEL / STOP (จุดต่อจุด) ---
        p_next = np.array([waypoints[i+1]['x'], waypoints[i+1]['y'], waypoints[i+1]['z']])
        
        # สร้าง Trajectory เดินทาง (หยุดหัว หยุดท้ายแน่นอน)
        pos, vel, acc = generate_travel_segment(p_curr, p_next, DT, SPEED_TRAVEL)
        
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
print(f"Done! Saved to {OUTPUT_CSV}")

# === VISUALIZATION ===
fig = plt.figure(figsize=(12, 6))

# Plot 1: 3D Path
ax1 = fig.add_subplot(121, projection='3d')
plot_df = df_final.iloc[::5]
colors = ['green' if t == 1 else 'red' for t in plot_df['type']]
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1)
ax1.set_title("Hybrid Path (Stop & Go)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# Plot 2: Velocity Profile
ax2 = fig.add_subplot(122)
speed = np.sqrt(df_final['vx']**2 + df_final['vy']**2 + df_final['vz']**2)
ax2.plot(df_final['t'], speed, label='Speed', color='black', linewidth=1)

# Highlight Drawing
mask = df_final['type'] == 1
ax2.fill_between(df_final['t'], 0, np.max(speed), where=mask, color='green', alpha=0.2, label='Drawing')

# ขีดเส้น 0 ให้เห็นชัดๆ
ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3)

ax2.set_title("Velocity Profile (Notice drops to 0 between segments)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (m/s)")
ax2.legend()
ax2.grid(True)

plt.show()