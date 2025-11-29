import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import os

# --- CONFIGURATION ---
INPUT_CSV = "Waypoints_For_Ruckig.csv"
OUTPUT_CSV = "Final_Trajectory_Arch_Full2.csv"

DT = 0.008 
Z_SAFE = 0.07          
SPEED_DRAW = 0.02
SPEED_TRAVEL = 0.1

# --- HELPER FUNCTIONS ---

def generate_arch_path(p_start, p_end, z_peak, v_limit, dt):
    """Arch Path (Sine Wave Z + Quintic XY)"""
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    
    dist_xy = np.linalg.norm(p_end[:2] - p_start[:2])
    
    if dist_xy < 0.001: return None, None, None

    total_path_len = abs(z_peak - p_start[2]) + dist_xy + abs(z_peak - p_end[2])
    duration = max(total_path_len / v_limit, 0.5) 
    
    num_steps = int(np.ceil(duration / dt))
    t = np.linspace(0, 1, num_steps) 
    
    # Quintic s (0->1)
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    dds = 60*t - 180*t**2 + 120*t**3
    
    # XY Interpolation
    x = p_start[0] + (p_end[0] - p_start[0]) * s
    y = p_start[1] + (p_end[1] - p_start[1]) * s
    
    vx = (p_end[0] - p_start[0]) * ds / duration
    vy = (p_end[1] - p_start[1]) * ds / duration
    
    ax = (p_end[0] - p_start[0]) * dds / duration**2
    ay = (p_end[1] - p_start[1]) * dds / duration**2

    # Z Interpolation (Sine Wave)
    z_base = p_start[2]
    arch_height = z_peak - z_base
    z = z_base + arch_height * np.sin(np.pi * s)
    
    # Velocity Z (Chain Rule)
    vz = arch_height * (np.pi * np.cos(np.pi * s)) * ds / duration
    
    # Acceleration Z (Numerical Gradient is safer for complex sine chain rule)
    # แต่ถ้าจะเอาสูตรเป๊ะๆ: d(vz)/dt
    # vz = K * cos(pi*s) * ds
    # az = K * [ -pi*sin(pi*s)*ds*ds + cos(pi*s)*dds ] / duration
    K = arch_height * np.pi / duration
    az = K * (-np.pi * np.sin(np.pi * s) * ds * ds + np.cos(np.pi * s) * dds) / duration
    
    pos = np.vstack((x, y, z)).T
    vel = np.vstack((vx, vy, vz)).T
    acc = np.vstack((ax, ay, az)).T
    
    return pos, vel, acc

def generate_draw_segment(points, v_limit, dt):
    """Draw Path (Spline)"""
    points = np.array(points)
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.hstack(([True], diff > 1e-6))]
    
    if len(points) < 2: return None, None, None
    
    points[:, 2] = points[0, 2] # Flatten Z
    
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_dist = np.sum(dists)
    duration = max(total_dist / v_limit, 0.1)
    
    t_points = np.hstack(([0], np.cumsum(dists))) / total_dist * duration
    
    try:
        cs = CubicSpline(t_points, points, axis=0, bc_type='clamped')
        num_steps = max(int(np.ceil(duration / dt)), 2)
        t_eval = np.linspace(0, duration, num_steps)
        
        pos = cs(t_eval)
        pos[:, 2] = points[0, 2] # Force flat Z
        
        vel = cs(t_eval, nu=1)
        vel[:, 2] = 0 # Force Vz=0
        
        acc = cs(t_eval, nu=2)
        acc[:, 2] = 0 # Force Az=0
        
        return pos, vel, acc
    except:
        return None, None, None

# --- MAIN LOOP ---
if not os.path.exists(INPUT_CSV):
    print("Error: Input file not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
waypoints = df.to_dict('records')

full_traj_rows = []
global_t = 0.0

current_pos = np.array([waypoints[0]['x'], waypoints[0]['y'], waypoints[0]['z']])

i = 1 
while i < len(waypoints):
    target = waypoints[i]
    
    if target['type'] == 0:
        # === ARCH ===
        target_pos = np.array([target['x'], target['y'], target['z']])
        pos, vel, acc = generate_arch_path(current_pos, target_pos, Z_SAFE, SPEED_TRAVEL, DT)
        
        if pos is not None:
            for j in range(len(pos)):
                full_traj_rows.append({
                    't': global_t, 
                    'x': pos[j,0], 'y': pos[j,1], 'z': pos[j,2], 
                    'vx': vel[j,0], 'vy': vel[j,1], 'vz': vel[j,2], 
                    'ax': acc[j,0], 'ay': acc[j,1], 'az': acc[j,2], # <--- เพิ่มบรรทัดนี้
                    'type': 0
                })
                global_t += DT
            current_pos = target_pos
        i += 1
        
    elif target['type'] == 1:
        # === DRAW ===
        draw_chunk = [current_pos]
        k = i
        while k < len(waypoints) and waypoints[k]['type'] == 1:
            draw_chunk.append([waypoints[k]['x'], waypoints[k]['y'], waypoints[k]['z']])
            k += 1
            
        pos, vel, acc = generate_draw_segment(draw_chunk, SPEED_DRAW, DT)
        
        if pos is not None:
            for j in range(len(pos)):
                full_traj_rows.append({
                    't': global_t, 
                    'x': pos[j,0], 'y': pos[j,1], 'z': pos[j,2], 
                    'vx': vel[j,0], 'vy': vel[j,1], 'vz': vel[j,2], 
                    'ax': acc[j,0], 'ay': acc[j,1], 'az': acc[j,2], # <--- เพิ่มบรรทัดนี้
                    'type': 1
                })
                global_t += DT
            current_pos = pos[-1]
        i = k

# --- SAVE & PLOT ---
df_final = pd.DataFrame(full_traj_rows)
# จัดลำดับ Column ให้สวยงาม
cols = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'type']
df_final = df_final[cols]

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Saved {len(df_final)} points with acceleration data.")

# Check Graphs
fig = plt.figure(figsize=(14, 8))

# 1. Z Profile
ax1 = fig.add_subplot(221)
ax1.plot(df_final['t'], df_final['z'], 'b-')
ax1.set_title("Z Position")
ax1.grid(True)

# 2. Acceleration Magnitude (Check Jerk)
ax2 = fig.add_subplot(222)
acc_mag = np.sqrt(df_final['ax']**2 + df_final['ay']**2 + df_final['az']**2)
ax2.plot(df_final['t'], acc_mag, 'r-')
ax2.set_title("Acceleration Magnitude")
ax2.grid(True)

# 3. 3D Path
ax3 = fig.add_subplot(212, projection='3d')
colors = ['red' if t==0 else 'green' for t in df_final['type']]
ax3.scatter(df_final['x'][::10], df_final['y'][::10], df_final['z'][::10], c=colors[::10], s=1)
ax3.set_title("3D Path")
ax3.set_box_aspect([1,1,0.5])

plt.tight_layout()
plt.show()