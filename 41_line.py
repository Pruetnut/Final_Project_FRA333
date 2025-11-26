import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
dt = 0.008            # Time step (UR5)
V_DRAW = 0.10         # Drawing speed (m/s)
Z_UP = 0.05
Z_DOWN = 0.00

# ===============================
# 1. DEFINE STRAIGHT LINE POINTS
# ===============================
P_start = np.array([0.40, -0.20, Z_UP])
P_down  = np.array([0.40, -0.20, Z_DOWN])
P_end   = np.array([0.55, -0.20, Z_DOWN])
P_up    = np.array([0.55, -0.20, Z_UP])

# ===============================
# 2. FUNCTION: generate linear trajectory with velocity
# ===============================
def generate_linear_traj_with_velocity(P1, P2, speed, dt):
    dist = np.linalg.norm(P2 - P1)
    if dist < 1e-6:
        return np.array([[*P2, 0, 0, 0]])
    
    T = dist / speed
    N = max(int(T / dt), 2)
    t = np.linspace(0, T, N)
    
    # position
    traj = np.outer(1 - t/T, P1) + np.outer(t/T, P2)
    
    # velocity (constant)
    v = (P2 - P1) / T
    vel = np.tile(v, (N,1))
    
    # acceleration = 0 (constant velocity)
    acc = np.zeros_like(traj)
    
    # Combine [x,y,z,vx,vy,vz,ax,ay,az]
    traj_full = np.hstack([traj, vel, acc])
    return traj_full

# ===============================
# 3. BUILD FULL TRAJECTORY
# ===============================
traj_list = []

# move above start
traj_list.append(generate_linear_traj_with_velocity(P_start, P_down, V_DRAW, dt))
# draw line
traj_list.append(generate_linear_traj_with_velocity(P_down, P_end, V_DRAW, dt))
# lift pen
traj_list.append(generate_linear_traj_with_velocity(P_end, P_up, V_DRAW, dt))

traj_full = np.vstack(traj_list)

# ===============================
# 4. ADD TIME COLUMN
# ===============================
time = np.arange(len(traj_full)) * dt
traj_full = np.hstack([time.reshape(-1,1), traj_full])

# Save CSV
header = "t,x,y,z,vx,vy,vz,ax,ay,az"
np.savetxt("straight_line_traj_with_velocity.csv", traj_full, delimiter=",", header=header, comments='')
print("Trajectory saved → straight_line_traj_with_velocity.csv")
print("Total points:", len(traj_full))

# ===============================
# 5. PLOT TRAJECTORY
# ===============================
plt.figure(figsize=(8,5))
plt.plot(traj_full[:,1], traj_full[:,2], '.-', markersize=2)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Straight Line Trajectory (XY)")
plt.axis("equal")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(traj_full[:,0], traj_full[:,3], label="Z height")
plt.xlabel("Time (s)")
plt.ylabel("Z (m)")
plt.title("Pen Up/Down")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(traj_full[:,0], traj_full[:,4], label="VX")
plt.plot(traj_full[:,0], traj_full[:,5], label="VY")
plt.plot(traj_full[:,0], traj_full[:,6], label="VZ")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Profile")
plt.legend()
plt.grid(True)
plt.show()
