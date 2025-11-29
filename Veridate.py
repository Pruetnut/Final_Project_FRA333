import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- CONFIGURATION ---
# ใส่ชื่อไฟล์ Trajectory ที่คุณต้องการตรวจสอบตรงนี้
INPUT_CSV = "Final_Trajectory_02.csv"

# --- LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: File '{INPUT_CSV}' not found. Please run the trajectory generation script first.")
    exit()

print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} data points.")

# --- VISUALIZATION DASHBOARD ---
# ตั้งค่าขนาดรูปภาพ (Full HD size)
fig = plt.figure(figsize=(18, 12))
plt.suptitle(f"Trajectory Analysis Dashboard: {INPUT_CSV}", fontsize=16, fontweight='bold')

# 1. 3D Path (Isometric View)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
# Downsample: พลอตทุกๆ 5 จุด เพื่อความเร็ว (ข้อมูลจริงยังอยู่ครบ)
plot_df = df.iloc[::5]
colors = ['red' if t == 0 else 'green' for t in plot_df['type']]
ax1.scatter(plot_df['x'], plot_df['y'], plot_df['z'], c=colors, s=1, alpha=0.5)
ax1.set_title("1. 3D Trajectory Path\n(Red=Travel/Lift, Green=Draw)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
# บังคับสัดส่วนแกน Z ให้ดูง่าย (ยืดแกน Z ออก)
ax1.set_box_aspect([1, 1, 0.5])

# 2. XY Plane (Top View) - เช็คความถูกต้องของรูปวาด
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(plot_df['x'], plot_df['y'], c=colors, s=1, alpha=0.5)
ax2.set_title("2. Top View (XY Plane)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.axis('equal') # สัดส่วนจริง
ax2.grid(True, linestyle='--', alpha=0.5)

# 3. Z-Height Profile - เช็คการยกปากกา
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(df['t'], df['z'], 'b-', linewidth=1.5, label='Z Height')
ax3.set_title("3. Z-Height Profile (Check Lifts)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Z (m)")
ax3.grid(True)
ax3.legend()

# 4. Velocity Magnitude - เช็คความเร็วรวม
ax4 = fig.add_subplot(2, 3, 4)
# คำนวณความเร็วรวม (Vector Norm)
speed = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
ax4.plot(df['t'], speed, 'k-', linewidth=1, label='|V|')
# ถมสีเขียวช่วงที่กำลังวาด
ax4.fill_between(df['t'], 0, speed.max(), where=df['type']==1, color='green', alpha=0.2, label='Drawing Phase')
ax4.set_title("4. Speed Profile (|V|)")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Speed (m/s)")
ax4.legend(loc='upper right')
ax4.grid(True)

# 5. Velocity Components - เช็คความเร็วแยกแกน
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(df['t'], df['vx'], label='Vx', alpha=0.7, linewidth=1)
ax5.plot(df['t'], df['vy'], label='Vy', alpha=0.7, linewidth=1)
ax5.plot(df['t'], df['vz'], label='Vz', alpha=0.7, linewidth=1)
ax5.set_title("5. Velocity Components (XYZ)")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Velocity (m/s)")
ax5.legend(loc='upper right')
ax5.grid(True)

# 6. Acceleration Magnitude - เช็คแรงกระชาก
ax6 = fig.add_subplot(2, 3, 6)
accel = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
ax6.plot(df['t'], accel, 'r-', linewidth=1, label='|A|')
ax6.set_title("6. Acceleration Magnitude (|A|)")
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Accel (m/s^2)")
ax6.grid(True)

# จัดระยะห่างกราฟให้สวยงาม
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# แสดงผล
print("Displaying dashboard...")
plt.show()