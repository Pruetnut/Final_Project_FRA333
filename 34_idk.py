import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# -------------------- PARAMETERS --------------------
image_path = "image/FIBO.png"      
# image_path = "image/Bird.JPG"
# image_path = "image/Maps.JPG"
outputname = "trajectory_ready_for_IK.csv"

pixel_size = 0.001                  # meters per pixel (ตัวอย่าง: 1 mm/pixel) -> ปรับให้ตรงกับภาพจริง
min_contour_len_px = 20            # ตัด contour สั้นๆ ทิ้ง (พิกเซล)
resample_ds = 0.002                # resample spacing in meters (2 mm)
z_down = 0.01                       # pen-down Z (m)
z_up = 0.05                        # pen-up Z (m)
# UR5-like limits (ปรับตามสเปกจริงของหุ่นคุณ)
vmax_tcp = 0.5                     # m/s (TCP max speed) - ปรับเป็นค่าจริงของ UR5
amax_tcp = 1.0                     # m/s^2 (max linear accel)
pen_down_speed_factor = 0.7        # ใช้ 70% ของ vmax/amax ตอนวาด

# Canny params
canny_thresh1 = 100
canny_thresh2 = 300
gauss_ksize = 5  # Gaussian blur kernel size

# ----------------------------------------------------

def pixel_coords_to_metric(contour, h, pixel_size):
    # OpenCV contour coordinates are (x_col, y_row) with y downwards.
    # Convert to (x, y) meters with y upward by flipping row coordinate.
    pts_px = contour.reshape(-1, 2)  # Nx2: (col, row)
    x = pts_px[:, 0] * pixel_size
    y = (h - pts_px[:, 1]) * pixel_size
    return np.column_stack((x, y))

def resample_path_by_arclen(pts, ds):
    # pts: Nx2 in meters, ordered
    if pts.shape[0] < 2:
        return pts
    seglen = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.hstack(([0.0], np.cumsum(seglen)))
    L = s[-1]
    if L == 0:
        return pts
    s_new = np.arange(0, L + 1e-9, ds)
    fx = interpolate.interp1d(s, pts[:,0], kind='linear')
    fy = interpolate.interp1d(s, pts[:,1], kind='linear')
    xq = fx(s_new)
    yq = fy(s_new)
    return np.column_stack((xq, yq)), s_new

def trapezoidal_profile_for_length(L, vmax, amax, ds):
    """Return arrays s_profile (arc-length samples), t_profile (time stamps) for a motion along length L."""
    if L <= 0:
        return np.array([0.0]), np.array([0.0])
    # time to accelerate to vmax
    ta = vmax / amax
    da = 0.5 * amax * ta**2
    if 2*da >= L:
        # triangular profile: peak velocity vm
        vm = np.sqrt(L * amax)
        ta = vm / amax
        tc = 0
        t_acc = np.linspace(0, ta, max(2, int(np.ceil(ta / (ds / vm)))))
        # generate s(t) numeric with small dt
        dt = min(ta/50.0, 0.01)
        times = np.arange(0, 2*ta+dt, dt)
        s = np.zeros_like(times)
        for i, tt in enumerate(times):
            if tt <= ta:
                s[i] = 0.5 * amax * tt**2
            else:
                td = tt - ta
                s[i] = (0.5 * amax * ta**2) + vm*td - 0.5*amax*td**2
        return s, times
    else:
        # trapezoidal
        tc = (L - 2*da) / vmax  # cruise time
        T = 2*ta + tc
        # choose dt so that s resolution ~ ds
        dt = min(ta/50.0, max(0.001, ds / vmax * 0.5))
        times = np.arange(0, T+dt, dt)
        s = np.zeros_like(times)
        for i, tt in enumerate(times):
            if tt <= ta:
                s[i] = 0.5 * amax * tt**2
            elif tt <= ta + tc:
                s[i] = da + vmax * (tt - ta)
            else:
                td = tt - (ta + tc)
                s[i] = da + vmax*tc + vmax*td - 0.5*amax*td**2
        return s, times

# -------------------- Read image & detect edges --------------------
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Cannot open {image_path}")

h, w = img.shape
# blur then Canny
img_blur = cv2.GaussianBlur(img, (gauss_ksize, gauss_ksize), 0)
edges = cv2.Canny(img_blur, canny_thresh1, canny_thresh2)

# find contours (ordered chains)
contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

# -------------------- Extract strokes (contours) --------------------
strokes = []
for cnt in contours:
    if len(cnt) < min_contour_len_px:
        continue
    pts_m = pixel_coords_to_metric(cnt, h, pixel_size)  # Nx2 meters
    # optionally reverse order to minimize jumps: we'll keep as is
    strokes.append(pts_m)

# sort strokes by their centroid (optional heuristic) to get consistent ordering
strokes = sorted(strokes, key=lambda s: np.mean(s[:,0]) + np.mean(s[:,1]))

# -------------------- Resample strokes uniformly by arc-length --------------------
resampled_strokes = []
for st in strokes:
    pts_res, s_arr = resample_path_by_arclen(st, resample_ds)
    resampled_strokes.append(pts_res)

# -------------------- Build full trajectory with pen-up / pen-down and time-profile --------------------
traj_points = []   # list of (x,y,z,t) will be filled
traj_v = []        # velocity along path (m/s)
traj_a = []        # acceleration along path (m/s^2)

global_time = 0.0

for i, stroke in enumerate(resampled_strokes):
    # pen down stroke: use reduced speed/accel
    L = np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1)).sum() if len(stroke)>1 else 0.0
    vmax_use = vmax_tcp * pen_down_speed_factor
    amax_use = amax_tcp * pen_down_speed_factor
    # create s,t profile
    s_profile, t_profile = trapezoidal_profile_for_length(L, vmax_use, amax_use, resample_ds)
    if len(s_profile) == 1 and L==0:
        # single point
        traj_points.append((stroke[0,0], stroke[0,1], z_down, global_time))
        traj_v.append(0.0); traj_a.append(0.0)
        global_time += 0.01
    else:
        # map s_profile (0..L) to stroke points using interp along arc-length
        # compute cumulative arc-length of stroke
        segs = np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1))
        s_cum = np.hstack(([0.0], np.cumsum(segs)))
        fx = interpolate.interp1d(s_cum, stroke[:,0], kind='linear', bounds_error=False, fill_value=(stroke[0,0], stroke[-1,0]))
        fy = interpolate.interp1d(s_cum, stroke[:,1], kind='linear', bounds_error=False, fill_value=(stroke[0,1], stroke[-1,1]))
        # for each sample in s_profile, compute x,y at time t_profile
        for s_val, t_rel in zip(s_profile, t_profile):
            x = float(fx(s_val))
            y = float(fy(s_val))
            traj_points.append((x, y, z_down, global_time + t_rel))
            # approximate v and a numerically (here we use derivative of s_profile)
            # compute local v and a from numeric derivative of s_profile vs time
        # after stroke, advance global_time
        global_time += t_profile[-1] + 0.01  # small dwell

    # If there is a next stroke, add pen-up travel from end of this stroke to start of next
    if i < len(resampled_strokes)-1:
        end_pt = stroke[-1]
        next_start = resampled_strokes[i+1][0]
        # linear travel path (pen up) from end_pt to next_start
        vec = next_start - end_pt
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            continue
        # use full vmax and amax for travel
        s_profile2, t_profile2 = trapezoidal_profile_for_length(dist, vmax_tcp, amax_tcp, resample_ds)
        # param along line
        for s_val, t_rel in zip(s_profile2, t_profile2):
            alpha = s_val / dist
            x = float(end_pt[0] + alpha * vec[0])
            y = float(end_pt[1] + alpha * vec[1])
            traj_points.append((x, y, z_up, global_time + t_rel))
        global_time += t_profile2[-1] + 0.01

# -------------------- Convert traj_points list to array and compute velocities/accelerations --------------------
traj = np.array(traj_points)  # columns: x,y,z,t
if traj.size == 0:
    raise RuntimeError("No trajectory generated (no contours found).")
# sort by time (just in case)
traj = traj[traj[:,3].argsort()]

times = traj[:,3]
pos = traj[:,0:3]

# compute numeric velocity and acceleration (finite differences)
dt = np.diff(times, prepend=times[0])
vel = np.zeros_like(pos)
acc = np.zeros_like(pos)
for k in range(1, len(times)):
    dtk = times[k] - times[k-1] if times[k] - times[k-1] > 1e-9 else 1e-9
    vel[k,:] = (pos[k,:] - pos[k-1,:]) / dtk
for k in range(1, len(times)):
    dtk = times[k] - times[k-1] if times[k] - times[k-1] > 1e-9 else 1e-9
    acc[k,:] = (vel[k,:] - vel[k-1,:]) / dtk

speed = np.linalg.norm(vel, axis=1)
accel = np.linalg.norm(acc, axis=1)

# -------------------- Save CSV ready for IK (x,y,z,t,v) --------------------
out = np.column_stack((times, pos[:,0], pos[:,1], pos[:,2], speed, accel))
header = "t,x,y,z,v,accel"
np.savetxt(outputname, out, delimiter=",", header=header, comments='')
print("Saved trajectory_ready_for_IK.csv with {} points".format(len(out)))

# -------------------- PLOTS --------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
# XY plot: draw pen-down in blue, pen-up in red
# Determine pen state by z
pen_down_mask = pos[:,2] <= (z_down + 1e-6)
plt.plot(pos[pen_down_mask,0], pos[pen_down_mask,1], '-b', linewidth=1)
plt.plot(pos[~pen_down_mask,0], pos[~pen_down_mask,1], '--r', linewidth=1)
plt.title("XY Path (blue=pen-down, red=pen-up)")
plt.axis('equal')

plt.subplot(1,2,2)
plt.plot(times, pos[:,2], '-k')
plt.title("Z (pen) vs time")
plt.xlabel("time (s)")
plt.ylabel("Z (m)")

plt.tight_layout()
plt.show()

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], '-k')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('3D trajectory (task space)')
plt.show()
