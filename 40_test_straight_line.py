import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================

dt = 0.008              # UR5 controller sampling time
V_DRAW = 0.10           # drawing speed (m/s)
Z_UP = 0.05             # pen up height
Z_DOWN = 0.00           # pen contact height

# ===============================
# 1. DEFINE STRAIGHT LINE POINTS
# ===============================

# Start & End (in robot base frame)
P_start = np.array([0.40, -0.20, Z_UP])     # move above start
P_down  = np.array([0.40, -0.20, Z_DOWN])   # press pen
P_end   = np.array([0.55, -0.20, Z_DOWN])   # draw to here
P_up    = np.array([0.55, -0.20, Z_UP])     # lift pen

# ===============================
# 2. FUNCTION: generate linear path
# ===============================

def generate_linear_traj(P1, P2, speed, dt):
    dist = np.linalg.norm(P2 - P1)
    T = dist / speed                # required time
    N = int(T / dt)                 # number of samples
    t = np.linspace(0, 1, N)

    traj = (1 - t)[:, None] * P1 + t[:, None] * P2
    return traj

# ===============================
# 3. BUILD TRAJECTORY PIPELINE
# ===============================

traj = []

# move above start
traj.append(generate_linear_traj(P_start, P_down, V_DRAW, dt))

# draw straight line
traj.append(generate_linear_traj(P_down, P_end, V_DRAW, dt))

# lift pen
traj.append(generate_linear_traj(P_end, P_up, V_DRAW, dt))

traj_full = np.vstack(traj)

# ===============================
# 4. EXPORT CSV (Optional)
# ===============================

np.savetxt("straight_line_traj.csv", traj_full,
           delimiter=",", header="x,y,z", comments="")

print("Trajectory saved → straight_line_traj.csv")
print("Total points:", len(traj_full))

# ===============================
# 5. VISUALIZE
# ===============================

plt.figure(figsize=(8,5))
plt.plot(traj_full[:,0], traj_full[:,1], '.-', markersize=2)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Straight Line Trajectory (XY)")
plt.axis("equal")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(traj_full[:,2], label="Z height")
plt.xlabel("Sample")
plt.ylabel("Z (m)")
plt.title("Pen Up/Down")
plt.grid(True)
plt.show()
