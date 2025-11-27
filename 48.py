"""
ur5_toppra_trajectory.py
- Input: image -> edges -> contours -> strokes (same approach you already had)
- For each stroke: fit spline geometry, then time-parameterize with toppra (v_max, a_max)
- Output CSV: t,x,y,z,vx,vy,vz,ax,ay,az
- Requires: numpy, scipy, opencv-python, pandas, matplotlib, toppra
"""

import os
import sys
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d

# toppra imports (must have toppra installed)
try:
    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algo
except Exception as e:
    raise ImportError("toppra is required. Install with `pip install toppra`. "
                      "Error: " + str(e))

# -------------------------
# 1) CONFIGURATION
# -------------------------
DRAW_MODE = "FLOOR"         # "FLOOR" or "WALL"
IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = f"ur5_toppra_{DRAW_MODE.lower()}.csv"

# image->physical mapping
CANVAS_WIDTH_M = 0.30       # physical width (m) that the image maps to
IMG_PROCESS_WIDTH = 600     # pixels (resized width)
MIN_CONTOUR_LEN = 12
VIA_POINT_DIST = 0.005      # 5 mm

# pen & plane offsets
BASE_TO_PLANE_DIST = 0.20   # your requested 20 cm from robot base to drawing plane
OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0

PEN_OFFSET_DOWN = 0.0
PEN_SAFE_UP = 0.05

# motion limits (tune these)
V_MAX_DRAW = 0.04          # m/s for drawing
A_MAX_DRAW = 0.5           # m/s^2
V_MAX_TRAVEL = 0.10        # m/s for travel (pen-up)
A_MAX_TRAVEL = 0.8         # m/s^2

UR5_DT = 0.008             # sampling time for output

PLOT_DEBUG = True

# Derived plane variables
if DRAW_MODE == "WALL":
    # wall: plane is Y-Z, X is depth
    PLANE_X = BASE_TO_PLANE_DIST + OFFSET_X
    START_POS_H = OFFSET_Y
    START_POS_V = OFFSET_Z
else:
    # floor: plane is X-Y, Z is height
    PLANE_Z = BASE_TO_PLANE_DIST + OFFSET_Z
    START_POS_H = OFFSET_Y
    START_POS_V = OFFSET_X

# -------------------------
# 2) Helper functions
# -------------------------
def process_image_to_edges(image_path, target_width=IMG_PROCESS_WIDTH):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found: " + image_path)
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h))
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 9, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    return edges, new_h, target_width

def downsample_points(points, min_dist=VIA_POINT_DIST):
    if len(points) < 2:
        return np.array(points)
    kept = [points[0]]
    last = points[0]
    for p in points[1:]:
        if np.linalg.norm(p - last) >= min_dist:
            kept.append(p)
            last = p
    if not np.allclose(kept[-1], points[-1]):
        kept.append(points[-1])
    return np.array(kept)

def order_strokes_greedy(strokes, home):
    """Greedy nearest-neighbour order of strokes for shorter travel"""
    remaining = strokes.copy()
    ordered = []
    cur = home
    while remaining:
        best_i = None
        best_d = float("inf")
        best_rev = False
        for i,s in enumerate(remaining):
            d0 = np.linalg.norm(s[0] - cur)
            d1 = np.linalg.norm(s[-1] - cur)
            if d0 < best_d:
                best_d = d0; best_i = i; best_rev = False
            if d1 < best_d:
                best_d = d1; best_i = i; best_rev = True
        chosen = remaining.pop(best_i)
        if best_rev:
            chosen = chosen[::-1]
        ordered.append(chosen)
        cur = chosen[-1]
    return ordered

# -------------------------
# 3) TOPPRA wrapper
# -------------------------
def generate_toppra_trajectory(points, vmax, amax, dt, start_t=0.0, path_smooth_s=1e-4):
    """
    Input:
      - points: Nx3 numpy array (Cartesian path)
      - vmax: scalar maximum cartesian speed (applied per-axis equally)
      - amax: scalar maximum cartesian accel
      - dt: desired sampling time for output
      - start_t: start time offset
    Output:
      - rows: list of [t,x,y,z,vx,vy,vz,ax,ay,az]
      - end_time
    Notes:
      - Uses toppra.SplineInterpolator and TOPPRA solver (seidel recommended).
      - We treat the 3D geometric path as a 3-DOF "joint" path and apply joint-like limits per axis.
    """
    pts = np.asarray(points)
    if pts.shape[0] < 2:
        return [], start_t

    # Prepare SplineInterpolator: nodes parameter array must be monotonic in [0,1]
    u_nodes = np.linspace(0.0, 1.0, pts.shape[0])
    path = ta.SplineInterpolator(u_nodes, pts.T.tolist())  # SplineInterpolator expects (u_list, waypoints_list)
    # Setup constraints (apply same scalar limit per axis)
    vel_limit = np.array([vmax]*3)
    acc_limit = np.array([amax]*3)
    vel_constr = constraint.JointVelocityConstraint(vel_limit)
    acc_constr = constraint.JointAccelerationConstraint(acc_limit)

    constraints = [vel_constr, acc_constr]

    # Solve TOPP-RA time-optimal parameterization
    instance = algo.TOPPRA(constraints, path, solver_wrapper='seidel')
    # compute_trajectory() returns a trajectory object (piecewise poly)
    traj = instance.compute_trajectory()
    T = traj.get_duration()

    # sample at dt
    t_samples = np.arange(0.0, T + 1e-9, dt)
    # trajectory evaluation
    pos = np.array(traj.eval(t_samples))   # shape (len, dof)
    vel = np.array(traj.evald(t_samples))
    acc = np.array(traj.evaldd(t_samples))

    rows = []
    for i, tt in enumerate(t_samples):
        rows.append([start_t + tt,
                     pos[i,0], pos[i,1], pos[i,2],
                     vel[i,0], vel[i,1], vel[i,2],
                     acc[i,0], acc[i,1], acc[i,2]])
    return rows, start_t + T

# -------------------------
# 4) Pen-up travel (simple cubic, time-limited)
# -------------------------
def compute_cubic_segment(p0, p1, vmax, amax, dt, start_t):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-9:
        return [], start_t
    T_vel = L / max(vmax, 1e-9)
    T_acc = math.sqrt(6 * L / max(amax, 1e-9))
    T = max(T_vel, T_acc, dt)
    a0 = p0
    a1 = np.zeros(3)
    a2 = (3*(p1-p0)/T**2) - (2*a1)/T
    a3 = (2*(p0-p1)/T**3) + (a1)/T**2
    rows = []
    t = start_t
    n_steps = max(int(math.ceil(T/dt)), 1)
    for i in range(n_steps):
        tau = i*dt
        pos = a0 + a1*tau + a2*tau**2 + a3*tau**3
        vel = a1 + 2*a2*tau + 3*a3*tau**2
        acc = 2*a2 + 6*a3*tau
        rows.append([t, pos[0], pos[1], pos[2], vel[0],vel[1],vel[2], acc[0],acc[1],acc[2]])
        t += dt
    rows.append([t, p1[0],p1[1],p1[2], 0.0,0.0,0.0, 0.0,0.0,0.0])
    return rows, t

def sanitize_stroke(points):
    """Remove duplicate consecutive points and ensure at least 3 valid points."""
    pts = np.asarray(points)
    # remove consecutive duplicates
    diff = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.insert(diff > 1e-9, 0, True)
    pts_clean = pts[keep]

    # if too short, return empty
    if pts_clean.shape[0] < 3:
        return np.array([])

    return pts_clean


# -------------------------
# 5) MAIN pipeline
# -------------------------
def main():
    print("1) Image processing...")
    edges, img_h, img_w = process_image_to_edges(IMAGE_PATH, IMG_PROCESS_WIDTH)
    cv2.imwrite("debug_edges.png", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("Contours:", len(contours))

    # Map contours to workspace points (meters)
    scale = CANVAS_WIDTH_M / img_w
    strokes = []
    for cnt in contours:
        if len(cnt) < MIN_CONTOUR_LEN:
            continue
        pts_px = cnt.reshape(-1,2).astype(float)
        if DRAW_MODE == "WALL":
            y_rob = pts_px[:,0]*scale + START_POS_H
            z_rob = (img_h - pts_px[:,1])*scale + START_POS_V
            x_rob = np.full_like(y_rob, PLANE_X + PEN_OFFSET_DOWN)
            dense = np.column_stack([x_rob + OFFSET_X, y_rob + OFFSET_Y, z_rob + OFFSET_Z])
        else:
            y_rob = pts_px[:,0]*scale + START_POS_H
            x_rob = START_POS_V + (pts_px[:,1]*scale)
            z_rob = np.full_like(y_rob, PLANE_Z + PEN_OFFSET_DOWN)
            dense = np.column_stack([x_rob + OFFSET_X, y_rob + OFFSET_Y, z_rob + OFFSET_Z])
        via = downsample_points(dense, VIA_POINT_DIST)
        if via.shape[0] >= 2:
            strokes.append(via)

    if not strokes:
        print("No strokes found — adjust image/thresholds.")
        return

    # order strokes
    if DRAW_MODE == "WALL":
        home = np.array([PLANE_X, START_POS_H, START_POS_V])
    else:
        home = np.array([START_POS_V, START_POS_H, PLANE_Z])
    strokes_ordered = order_strokes_greedy(strokes, home)
    print("Strokes after ordering:", len(strokes_ordered))

    # build full trajectory using toppra for drawing segments
    full_rows = []
    t_now = 0.0
    last_pos = None

    for stroke in strokes_ordered:
        stroke = sanitize_stroke(stroke)
        if len(stroke) < 3:
            continue  # skip useless tiny strokes

        draw_rows, t_now = generate_toppra_trajectory(stroke, V_MAX_DRAW, A_MAX_DRAW, UR5_DT, t_now)


    # build DataFrame and save CSV
    df = pd.DataFrame(full_rows, columns=['t','x','y','z','vx','vy','vz','ax','ay','az'])
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
    print("Saved:", OUTPUT_CSV)

    # debug plots
    if PLOT_DEBUG:
        vnorm = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.plot(df['t'], vnorm)
        plt.title('Speed profile')
        plt.grid(True)
        ax = plt.subplot(212, projection='3d')
        ax.plot(df['x'], df['y'], df['z'], linewidth=0.6)
        ax.set_title('Trajectory (3D)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
