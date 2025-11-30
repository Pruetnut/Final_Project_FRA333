import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_CSV = "Matlab/f_Trajectory.csv"
ROWS_TO_PLOT = 3000  # เลือกจำนวน Row ที่ต้องการ (50 จุดแรก)

# --- LOAD DATA ---
if not os.path.exists(INPUT_CSV):
    print(f"Error: File '{INPUT_CSV}' not found.")
    # สร้าง Dummy data เพื่อทดสอบโค้ดกรณีไม่มีไฟล์จริง
    t = [i * 0.008 for i in range(100)]
    x = [0 if i < 10 else 0.05 for i in range(100)] # สมมติยกขึ้น
    vx = [0] * 100
    ax = [0] * 100
    df = pd.DataFrame({'t': t, 'x': x, 'vx': vx, 'ax': ax})
    print("Using dummy data for demonstration...")
else:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

# --- SELECT SUBSET ---
# เลือกแค่ 50 แถวแรก
df_subset = df.head(ROWS_TO_PLOT)

print(f"Plotting first {len(df_subset)} rows...")

# --- PLOTTING ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Z Position
ax1.plot(df_subset['t'], df_subset['x'], 'b.-', label='X Position')
ax1.set_ylabel('Position (m)')
ax1.set_title(f'X-Axis Profile (First {ROWS_TO_PLOT} points)')
ax1.grid(True)
ax1.legend()

# 2. Z Velocity
ax2.plot(df_subset['t'], df_subset['vx'], 'g.-', label='X Velocity')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('X-Axis Velocity')
ax2.grid(True)
ax2.legend()

# 3. Z Acceleration
ax3.plot(df_subset['t'], df_subset['ax'], 'r.-', label='X Acceleration')
ax3.set_ylabel('Acceleration (m/s^2)')
ax3.set_xlabel('Time (s)')
ax3.set_title('X-Axis Acceleration')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()