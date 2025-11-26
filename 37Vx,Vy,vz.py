import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# -------------------- PARAMETERS --------------------
image_path = "image/FIBO.png"
image_path = "image/line.png"
outputname = "trajectory_fixed_dt_ready_xyz.csv"


# Robot Control Parameters
UR5_DT = 0.02                     # Fixed sampling time (s)

pixel_size = 0.001                 # meters per pixel
min_contour_len_px = 20            # cut tiny contours
z_down = 0.01                      # pen-down Z (m)
z_up = 0.05                        # pen-up Z (m)

offset_x = 0.5   # ห่างจากฐานหุ่นไปด้านหน้า 50 cm
offset_y = 0.0   # ไม่เลื่อนด้านข้าง
offset_z = 0.2  # ยกสูงขึ้น 20 cm

# Kinematic Limits
vmax_tcp = 0.5                     # m/s
amax_tcp = 1.0                     # m/s^2
pen_down_speed_factor = 0.2        # use 70% when drawing

# Canny params
canny_thresh1 = 100
canny_thresh2 = 300
gauss_ksize = 5

# -------------------- HELPER FUNCTIONS --------------------
def pixel_coords_to_metric(contour, h, pixel_size):
    pts_px = contour.reshape(-1, 2)
    x = pts_px[:, 0] * pixel_size
    y = (h - pts_px[:, 1]) * pixel_size
    return np.column_stack((x, y))

def generate_trapezoidal_s_profile_fixed_dt(L, vmax, amax, dt):
    if L <= 1e-6:
        return np.array([0.0]), np.array([0.0])
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc**2
    if L < 2 * d_acc:
        # triangular
        t_acc = np.sqrt(L / amax)
        t_flat = 0.0
        v_peak = amax * t_acc
        total_time = 2 * t_acc
    else:
        d_flat = L - 2 * d_acc
        t_flat = d_flat / vmax
        v_peak = vmax
        total_time = 2 * t_acc + t_flat
    num_steps = int(np.ceil(total_time / dt))
    t_values = np.arange(0, num_steps + 1) * dt
    t_values = t_values[t_values <= total_time]
    if len(t_values) == 0 or t_values[-1] < total_time:
        t_values = np.append(t_values, total_time)
    s_values = np.zeros_like(t_values)
    for i, t in enumerate(t_values):
        if t <= t_acc:
            s_values[i] = 0.5 * amax * t**2
        elif t <= t_acc + t_flat:
            s_values[i] = d_acc + v_peak * (t - t_acc)
        else:
            t_dec = t - (t_acc + t_flat)
            s_values[i] = d_acc + (v_peak * t_flat) + (v_peak * t_dec) - (0.5 * amax * t_dec**2)
    s_values = np.clip(s_values, 0, L)
    return s_values, t_values

# -------------------- MAIN PROCESS --------------------

# 1. Image Processing
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Cannot open {image_path}")

h, w = img.shape
img_blur = cv2.GaussianBlur(img, (gauss_ksize, gauss_ksize), 0)
edges = cv2.Canny(img_blur, canny_thresh1, canny_thresh2)
contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

# 2. Extract Strokes & Sort
strokes = []
for cnt in contours:
    if len(cnt) < min_contour_len_px:
        continue
    pts_m = pixel_coords_to_metric(cnt, h, pixel_size)
    strokes.append(pts_m)

if not strokes:
    raise RuntimeError("No strokes extracted. Check image/parameters.")

# Simple heuristic sort (can be improved)
strokes = sorted(strokes, key=lambda s: (np.mean(s[:,0]) + np.mean(s[:,1])))

# 3. Build Trajectory with Fixed DT
traj_data = [] # will store rows: [t, x, y, z]
current_global_time = 0.0

print(f"Generating trajectory with fixed dt = {UR5_DT}s ...")

for i, stroke in enumerate(strokes):
    # compute arc-length and cumulative distances
    if stroke.shape[0] < 2:
        # single point
        traj_data.append([current_global_time, stroke[0,0], stroke[0,1], z_down])
        current_global_time += UR5_DT
        continue

    diffs = np.diff(stroke, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.hstack(([0.0], np.cumsum(seg_lengths)))
    L = cum_dist[-1]

    if L > 1e-6:
        interp_x = interpolate.interp1d(cum_dist, stroke[:,0], kind='linear', bounds_error=False, fill_value=(stroke[0,0], stroke[-1,0]))
        interp_y = interpolate.interp1d(cum_dist, stroke[:,1], kind='linear', bounds_error=False, fill_value=(stroke[0,1], stroke[-1,1]))

        v_use = vmax_tcp * pen_down_speed_factor
        a_use = amax_tcp * pen_down_speed_factor
        s_prof, t_prof = generate_trapezoidal_s_profile_fixed_dt(L, v_use, a_use, UR5_DT)

        # map local times to global times starting at current_global_time
        t_global = current_global_time + t_prof

        x_new = interp_x(s_prof)
        y_new = interp_y(s_prof)
        z_new = np.full_like(x_new, z_down)

        for j in range(len(t_global)):
            traj_data.append([t_global[j], float(x_new[j]), float(y_new[j]), float(z_new[j])])

        # advance time slightly to avoid duplicate time stamps
        current_global_time = t_global[-1] + UR5_DT * 0.5

    # Pen-up travel
    if i < len(strokes) - 1:
        start_pt = stroke[-1]
        end_pt = strokes[i+1][0]
        dist_travel = np.linalg.norm(end_pt - start_pt)
        if dist_travel > 1e-9:
            s_prof_up, t_prof_up = generate_trapezoidal_s_profile_fixed_dt(dist_travel, vmax_tcp, amax_tcp, UR5_DT)
            direction = (end_pt - start_pt) / dist_travel
            x_up = start_pt[0] + direction[0] * s_prof_up
            y_up = start_pt[1] + direction[1] * s_prof_up
            z_up_arr = np.full_like(x_up, z_up)
            t_global_up = current_global_time + t_prof_up
            for j in range(len(t_global_up)):
                traj_data.append([t_global_up[j], float(x_up[j]), float(y_up[j]), float(z_up_arr[j])])
            current_global_time = t_global_up[-1] + UR5_DT * 0.5

# Convert to numpy array and ensure sorted by time
traj_arr = np.array(traj_data)
traj_arr = traj_arr[traj_arr[:,0].argsort()]
# Apply offset to X, Y, Z
traj_arr[:,1] += offset_x
traj_arr[:,2] += offset_y
traj_arr[:,3] += offset_z
# ---------------------------------------------------------

# Save CSV
np.savetxt(outputname, traj_arr, delimiter=",",
           header="t, x, y, z", comments='')
print(f"Saved trajectory to {outputname}")


# 4. Compute per-axis velocity and acceleration (finite differences with fixed dt)
n = traj_arr.shape[0]
pos = traj_arr[:, 1:4]  # x,y,z
# initialize arrays
vel = np.zeros_like(pos)
acc = np.zeros_like(pos)

# Because we enforce fixed dt sampling in generation, we can use UR5_DT for diffs.
# However due to possible final time clipping we will compute dynamic dt between points but expect it ~ UR5_DT.
# Use forward difference for simplicity (and backward for last point).
for i in range(1, n):
    dt_local = traj_arr[i,0] - traj_arr[i-1,0]
    if dt_local <= 0:
        dt_local = UR5_DT  # fallback
    vel[i] = (pos[i] - pos[i-1]) / dt_local

for i in range(1, n):
    dt_local = traj_arr[i,0] - traj_arr[i-1,0]
    if dt_local <= 0:
        dt_local = UR5_DT
    acc[i] = (vel[i] - vel[i-1]) / dt_local

# optional: set first velocity/acc as same as second to avoid zero spike
if n > 1:
    vel[0] = vel[1]
    acc[0] = acc[1]

# Compute norms if you still want them
v_norm = np.linalg.norm(vel, axis=1)
a_norm = np.linalg.norm(acc, axis=1)

# Prepare CSV columns: t, x, y, z, vx, vy, vz, ax, ay, az
out_arr = np.column_stack((
    traj_arr[:,0],      # t
    pos[:,0], pos[:,1], pos[:,2],   # x,y,z
    vel[:,0], vel[:,1], vel[:,2],   # vx,vy,vz
    acc[:,0], acc[:,1], acc[:,2]    # ax,ay,az
))

header = "t,x,y,z,vx,vy,vz,ax,ay,az"
np.savetxt(outputname, out_arr, delimiter=",", header=header, comments='')

print(f"Saved {outputname}")
print(f"Total Points: {len(out_arr)}")
print(f"Total Time: {out_arr[-1,0]:.3f} s")
print(f"Fixed Time Step (expected): {UR5_DT} s")

# -------------------- PLOT --------------------
plt.figure(figsize=(10, 9))

plt.subplot(3,1,1)
plt.plot(pos[:,0], pos[:,1], '.-', markersize=1, linewidth=0.5)
plt.title(f"Trajectory Path (fixed dt ≈ {UR5_DT}s)")
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.subplot(3,1,2)
plt.plot(traj_arr[:,0], vel[:,0], label='vx')
plt.plot(traj_arr[:,0], vel[:,1], label='vy')
plt.plot(traj_arr[:,0], vel[:,2], label='vz')
plt.legend()
plt.title("Per-axis Velocity")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(traj_arr[:,0], acc[:,0], label='ax')
plt.plot(traj_arr[:,0], acc[:,1], label='ay')
plt.plot(traj_arr[:,0], acc[:,2], label='az')
plt.legend()
plt.title("Per-axis Acceleration")
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.grid(True)

plt.tight_layout()
plt.show()

