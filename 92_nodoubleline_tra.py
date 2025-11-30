import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Helper: quintic interpolation (velocity & acceleration = 0)
# -------------------------------------------------------------
def quintic_interpolate(p0, p1, n_points):
    t = np.linspace(0, 1, n_points)
    t = t.reshape(-1, 1)   # reshape เป็น (N,1)

    p0 = np.array(p0).reshape(1, 3)   # (1,3)
    p1 = np.array(p1).reshape(1, 3)   # (1,3)

    # คำนวณ quintic vector ทุกแกนพร้อมกัน
    h = (1 - 10*t**3 + 15*t**4 - 6*t**5)
    k = (10*t**3 - 15*t**4 + 6*t**5)

    traj = p0 * h + p1 * k   # ผลลัพธ์ shape = (N,3)
    return traj


# -------------------------------------------------------------
# Load CSV
# -------------------------------------------------------------
df = pd.read_csv("Waypoints_with_z4point.csv")

# Split by type
type0 = df[df["type"] == 0].reset_index(drop=True)
type1 = df[df["type"] == 1].reset_index(drop=True)

# Output arrays
traj_x = []
traj_y = []
traj_z = []
traj_type = []

# -------------------------------------------------------------
# 1) Process TYPE 0 → Use Quintic
# -------------------------------------------------------------
for i in range(len(type0)-1):
    p0 = type0.loc[i, ["x","y","z"]].values
    p1 = type0.loc[i+1, ["x","y","z"]].values

    N = 20  # number of interpolation points for smooth pen-up/pen-down
    q = quintic_interpolate(p0, p1, N)

    traj_x.extend(q[:,0])
    traj_y.extend(q[:,1])
    traj_z.extend(q[:,2])
    traj_type.extend([0]*N)

# -------------------------------------------------------------
# 2) Process TYPE 1 → Use Monotonic PCHIP
# -------------------------------------------------------------
points = type1[["x","y","z"]].values
dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
dist = np.insert(dist, 0, 0)    # arc-length

# Create monotonic interpolators
px = PchipInterpolator(dist, points[:,0])
py = PchipInterpolator(dist, points[:,1])
pz = PchipInterpolator(dist, points[:,2])

# Resample with smooth spacing
N = 800   # number of points on drawing lines
d_new = np.linspace(0, dist[-1], N)

traj_x.extend(px(d_new))
traj_y.extend(py(d_new))
traj_z.extend(pz(d_new))
traj_type.extend([1]*N)

# -------------------------------------------------------------
# Save cleaned trajectory
# -------------------------------------------------------------
clean_traj = pd.DataFrame({
    "x": traj_x,
    "y": traj_y,
    "z": traj_z,
    "type": traj_type
})
clean_traj.to_csv("clean_trajectory.csv", index=False)

print("Saved → clean_trajectory.csv")


# -------------------------------------------------------------
# Plot to confirm (no overlapping / no reverse)
# -------------------------------------------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj_x, traj_y, traj_z, linewidth=2)
ax.set_title("Cleaned Trajectory (No Overlap, No Reverse)")

plt.show()
