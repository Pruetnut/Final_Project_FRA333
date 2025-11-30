# trajectory_from_drawfile.py
# อ่าน CSV (x,y,z,type) -> แยก runs -> cubic/quintic spline -> sample dt=0.008s
# ต้องการ: pandas, numpy, matplotlib, scipy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BPoly
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3D plotting)

# --------------------
# User settings
# --------------------
csv_path = "Waypoints_with_z4point.csv"  
dt = 0.008                   # sampling time (s)
# UR5e (typical) TCP speed/acc limits (conservative). ปรับได้ตาม controller ของคุณ.
MAX_TCP_V = 1.0    # m/s (typical TCP speed per UR5e datasheet). :contentReference[oaicite:1]{index=1}
MAX_TCP_A = 1.0    # m/s^2 (conservative choice; ปรับได้)
# Minimum segment time to avoid degenerate zero-time segments
MIN_SEGMENT_TIME = 0.02  # s

# --------------------
# Utility functions
# --------------------
def find_runs(types):
    """Find contiguous runs of identical type.
    Return list of (start_idx, end_idx, type_val) inclusive indices.
    """
    runs = []
    if len(types) == 0:
        return runs
    start = 0
    cur = types[0]
    for i in range(1, len(types)):
        if types[i] != cur:
            runs.append((start, i-1, cur))
            start = i
            cur = types[i]
    runs.append((start, len(types)-1, cur))
    return runs

def compute_segment_durations(points, v_max=MAX_TCP_V, a_max=MAX_TCP_A):
    """
    Given points (Nx3), compute a set of knot times for the sequence.
    Strategy:
      - compute chord distances d_i between consecutive points
      - total length L = sum(d_i)
      - choose total_time = max(L / v_max, sqrt(6*L / a_max)) as heuristic
        (the sqrt(6L/a) comes from typical quintic/trapezoidal scaling to respect accel)
      - then distribute time across segments proportional to segment length
    Returns: t_knots (array length N) in seconds, starting at 0
    """
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    L = np.sum(seg_len)
    if L <= 0:
        # all points identical
        t_knots = np.zeros(points.shape[0])
        return t_knots

    # heuristic total time (guarantee no division by zero)
    t_by_v = L / max(1e-6, v_max)
    t_by_a = np.sqrt(max(1e-12, 6.0 * L / max(1e-6, a_max)))
    total_time = max(t_by_v, t_by_a, MIN_SEGMENT_TIME * len(seg_len))
    # distribute by length
    seg_time = np.zeros_like(seg_len)
    nonzero = seg_len > 0
    seg_time[nonzero] = (seg_len[nonzero] / np.sum(seg_len[nonzero])) * total_time
    # handle tiny/zero segments
    seg_time[~nonzero] = MIN_SEGMENT_TIME
    # build knot times
    t_knots = np.concatenate(([0.0], np.cumsum(seg_time)))
    return t_knots

# --------------------
# Main pipeline
# --------------------
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# Read CSV
df = pd.read_csv(csv_path)
# check columns
for col in ("x","y","z","type"):
    if col not in df.columns:
        raise ValueError(f"CSV ต้องมีคอลัมน์ '{col}'")

pts = df[["x","y","z"]].to_numpy(dtype=float)
types = df["type"].to_numpy()

runs = find_runs(types)

# storage for final sampled trajectory
time_all = []
pos_all = []
vel_all = []
acc_all = []

# process each run independently
t_offset = 0.0
for (sidx, eidx, tval) in runs:
    run_pts = pts[sidx:eidx+1, :]
    n_pts = run_pts.shape[0]

    # if only one point in run -> produce a tiny stop segment (stay still)
    if n_pts == 1:
        # stay at point for a single sample
        t_samples = np.array([t_offset, t_offset + dt])
        p_samples = np.tile(run_pts[0:1,:], (t_samples.size, 1))
        v_samples = np.zeros_like(p_samples)
        a_samples = np.zeros_like(p_samples)

        time_all.append(t_samples)
        pos_all.append(p_samples)
        vel_all.append(v_samples)
        acc_all.append(a_samples)
        t_offset = t_samples[-1]
        continue

    # compute knot times based on distances and UR5 constraints
    t_knots = compute_segment_durations(run_pts, v_max=MAX_TCP_V, a_max=MAX_TCP_A)
    # shift by t_offset so global time is continuous
    t_knots_global = t_knots + t_offset
    total_time = t_knots_global[-1] - t_knots_global[0]
    if total_time <= 0:
        total_time = max(MIN_SEGMENT_TIME, dt)

    # sample time vector for this run
    t_samples = np.arange(t_knots_global[0], t_knots_global[-1] + 1e-9, dt)

    if tval == 1:
        # ---------- cubic spline with zero end velocities ----------
        # We will build three scalar CubicSpline objects with clamped (zero derivative) ends.
        # bc_type = ((1, v0),(1, vN)) where (1, value) means first derivative fixed.
        cs_x = CubicSpline(t_knots_global, run_pts[:,0], bc_type=((1,0.0),(1,0.0)))
        cs_y = CubicSpline(t_knots_global, run_pts[:,1], bc_type=((1,0.0),(1,0.0)))
        cs_z = CubicSpline(t_knots_global, run_pts[:,2], bc_type=((1,0.0),(1,0.0)))

        p_samp = np.vstack((cs_x(t_samples), cs_y(t_samples), cs_z(t_samples))).T
        v_samp = np.vstack((cs_x(t_samples,1), cs_y(t_samples,1), cs_z(t_samples,1))).T
        a_samp = np.vstack((cs_x(t_samples,2), cs_y(t_samples,2), cs_z(t_samples,2))).T

    else:
        # ---------- quintic spline (end vel & acc = 0) ----------
        # Use BPoly.from_derivatives to create a piecewise polynomial that
        # enforces position + first and second derivatives at endpoints.
        # For interior knots we provide only positions (BPoly will determine internal coefficients).
        # Build y_list: at endpoints include [pos, vel, acc], interior only [pos]
        y_list = []
        for i in range(n_pts):
            p = run_pts[i,:]
            if i == 0 or i == (n_pts-1):
                # position + velocity (0) + acceleration (0)
                y_list.append([p, np.zeros(3), np.zeros(3)])
            else:
                y_list.append([p])
        # create BPoly (vector-valued) from derivatives
        bp = BPoly.from_derivatives(t_knots_global, y_list)
        # evaluate
        p_samp = bp(t_samples)
        v_samp = bp.derivative(1)(t_samples)
        a_samp = bp.derivative(2)(t_samples)

    # append
    time_all.append(t_samples)
    pos_all.append(p_samp)
    vel_all.append(v_samp)
    acc_all.append(a_samp)

    # update offset: ensure small gap so next run starts after previous (maintain start/stop)
    t_offset = t_samples[-1] + dt  # small pause between runs (allows velocities to be zero at boundary)

# concatenate all runs into single arrays
time_all = np.concatenate(time_all)
pos_all = np.vstack(pos_all)
vel_all = np.vstack(vel_all)
acc_all = np.vstack(acc_all)

# --------------------
# Sanity checks & clipping
# --------------------
# compute norms
vel_norm = np.linalg.norm(vel_all, axis=1)
acc_norm = np.linalg.norm(acc_all, axis=1)

# If any velocities or accelerations exceed limits, warn (but do not forcibly rescale here)
if np.max(vel_norm) > MAX_TCP_V * 1.0001:
    print("WARNING: max TCP velocity exceeds limit: {:.3f} m/s (limit {:.3f})".format(np.max(vel_norm), MAX_TCP_V))
if np.max(acc_norm) > MAX_TCP_A * 1.0001:
    print("WARNING: max TCP acceleration exceeds limit: {:.3f} m/s^2 (limit {:.3f})".format(np.max(acc_norm), MAX_TCP_A))

# --------------------
# Plot results
# --------------------
fig = plt.figure(figsize=(14,6))

# 3D path
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot(pos_all[:,0], pos_all[:,1], pos_all[:,2], '-', linewidth=1.5, label='trajectory')
# original waypoints for reference
ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c='r', s=20, label='waypoints')
ax1.set_title('3D Trajectory (sampled dt={:.3f}s)'.format(dt))
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.legend()
ax1.view_init(elev=30, azim=-60)
ax1.grid(True)

# velocity / acceleration norms
ax2 = fig.add_subplot(2,2,2)
ax2.plot(time_all, vel_norm, '-', label='|v| (m/s)')
ax2.axhline(MAX_TCP_V, linestyle='--', label='v limit')
ax2.set_xlabel('time (s)'); ax2.set_ylabel('speed (m/s)')
ax2.legend(); ax2.grid(True)

ax3 = fig.add_subplot(2,2,4)
ax3.plot(time_all, acc_norm, '-', label='|a| (m/s^2)')
ax3.axhline(MAX_TCP_A, linestyle='--', label='a limit')
ax3.set_xlabel('time (s)'); ax3.set_ylabel('accel (m/s^2)')
ax3.legend(); ax3.grid(True)

plt.tight_layout()
plt.show()

# --------------------
# Save sampled trajectory to CSV (optional)
# --------------------
out_df = pd.DataFrame({
    't': time_all,
    'x': pos_all[:,0], 'y': pos_all[:,1], 'z': pos_all[:,2],
    'vx': vel_all[:,0], 'vy': vel_all[:,1], 'vz': vel_all[:,2],
    'ax': acc_all[:,0], 'ay': acc_all[:,1], 'az': acc_all[:,2]
})
out_csv = "Matlab/sampled_trajectory.csv"
out_df.to_csv(out_csv, index=False)
print("Sampled trajectory saved to:", out_csv)
