#!/usr/bin/env python3
"""
image_to_trajectory.py

Input: IMAGE_PATH (png/jpg)
Output: CSV with columns: t,x,y,z,vx,vy,vz,ax,ay,az

Configurable parameters (see CONFIG section).

Author: ChatGPT (example)
"""

import cv2
import numpy as np
import pandas as pd
import math
import os
from pathlib import Path

# --------------------------
# CONFIG (ปรับค่าได้ตามต้องการ)
# --------------------------
IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = "trajectory_output.csv"

# Drawing / physical parameters (units: mm)
DRAWING_WIDTH_MM = 150.0      # ขนาดความกว้างของรูปที่วาด (mm)
ROBOT_ORIGIN_XY_MM = (0.0, 0.0)  # offset for robot workspace origin (mm)
PAPER_OFFSET_MM = (50.0, 50.0)   # additional translation offset to place drawing relative to robot origin

# Pen z-positions (mm)
DRAW_Z = 0.0     # z when drawing (touching paper)
LIFT_Z = 30.0    # z when pen lifted (above paper)

# Pen control timing
PEN_DOWN_DELAY = 0.10  # s : delay to wait after pen-down (you can insert extra frames)
PEN_UP_DELAY = 0.05    # s : delay to wait after pen-up

# Trajectory sampling & dynamics limits
SAMPLING_TIME = 0.02   # s (initial sampling dt)
MAX_VELOCITY = 200.0   # mm/s maximum linear speed desired (robot TCP speed)
MAX_ACCEL = 1000.0     # mm/s^2 maximum linear accel desired
PATH_POINT_SPACING = 1.0  # mm spacing when resampling paths (smaller = more points)

# Image processing
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
MIN_CONTOUR_LENGTH = 10  # ignore very small contours

# Output scaling / flip
FLIP_VERTICAL = True  # flip vertical axis (image coordinate to robot coordinate)
INVERT_COLORS = False  # set True if image is white-on-black vs black-on-white

# --------------------------
# Helpers
# --------------------------

def read_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img

def get_binary_edges(img):
    # optional invert
    if INVERT_COLORS:
        img = 255 - img
    # use Canny then dilate to connect small gaps optionally
    edges = cv2.Canny(img, CANNY_THRESH1, CANNY_THRESH2)
    # optionally dilate/erode to close small gaps - commented out for now
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # filter small
    filtered = [c.reshape(-1,2) for c in contours if len(c) >= MIN_CONTOUR_LENGTH]
    return filtered

def contour_length(pts):
    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    return d.sum()

def resample_path(pts, spacing=1.0):
    # pts: Nx2
    if len(pts) < 2:
        return pts.copy()
    # cumulative distance
    diffs = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = cum[-1]
    if total == 0:
        return pts[:1].copy()
    n_samples = max(1, int(math.ceil(total / spacing)))
    sample_d = np.linspace(0, total, n_samples+1)  # +1 to include endpoint
    new_pts = []
    for d in sample_d:
        # find segment
        idx = np.searchsorted(cum, d, side='right') - 1
        if idx >= len(pts)-1:
            new_pts.append(pts[-1])
        else:
            t = (d - cum[idx]) / (cum[idx+1] - cum[idx]) if (cum[idx+1] - cum[idx])>0 else 0.0
            p = (1-t)*pts[idx] + t*pts[idx+1]
            new_pts.append(p)
    return np.array(new_pts)

def image_points_to_world(pts, img_shape, drawing_width_mm, flip_vertical=True, origin_offset=(0,0)):
    """Map image coordinates (x col, y row) to world mm coordinates.
       - pts: Nx2 array in image coords (col, row)
       - img_shape: (h, w)
       - drawing_width_mm: desired width in mm
       returns Nx2 in mm"""
    h, w = img_shape
    scale = drawing_width_mm / float(w)
    # map
    xs = pts[:,0].astype(float) * scale
    ys = pts[:,1].astype(float) * scale
    if flip_vertical:
        ys = (h * scale) - ys
    # apply origin offset
    xs = xs + origin_offset[0]
    ys = ys + origin_offset[1]
    return np.column_stack([xs, ys])

def build_strokes_from_contours(contours, img_shape, cfg):
    strokes = []  # list of (x,y,pen) where pen=0 -> up, 1->down
    origin = (cfg['ROBOT_ORIGIN_XY_MM'][0] + cfg['PAPER_OFFSET_MM'][0],
              cfg['ROBOT_ORIGIN_XY_MM'][1] + cfg['PAPER_OFFSET_MM'][1])
    for i, c in enumerate(contours):
        res = resample_path(c, spacing=cfg['PATH_POINT_SPACING'])
        if len(res) < 2:
            continue
        world = image_points_to_world(res, img_shape, cfg['DRAWING_WIDTH_MM'], flip_vertical=cfg['FLIP_VERTICAL'], origin_offset=origin)
        # move to start (pen up), then pen down across points
        strokes.append((world[0,0], world[0,1], 0))  # pen up move-to
        for p in world:
            strokes.append((p[0], p[1], 1))  # pen down
        # After finishing contour, lift pen
        strokes.append((world[-1,0], world[-1,1], 0))
    return strokes

def insert_pen_delay_samples(samples, sampling_time, pen_down_delay, pen_up_delay):
    """samples: list of (x,y,z,pen). Insert additional samples (same pos) for pen delays when pen state changes.
       Return new list."""
    out = []
    prev_pen = None
    for x,y,z,pen in samples:
        if prev_pen is None:
            out.append((x,y,z,pen))
        else:
            if pen != prev_pen:
                # insert delay samples at previous position (or current) depending on transition
                delay = pen_down_delay if pen==1 else pen_up_delay
                if delay > 0 and sampling_time>0:
                    n_extra = int(math.ceil(delay / sampling_time))
                    for _ in range(n_extra):
                        # hold current position with same pen-state (the new state) to allow time for servo
                        out.append((x,y,z,pen))
                out.append((x,y,z,pen))
            else:
                out.append((x,y,z,pen))
        prev_pen = pen
    return out

def strokes_to_samples(strokes, draw_z, lift_z):
    """Convert strokes (x,y,pen) to samples (x,y,z,pen) with z assigned based on pen flag."""
    samples = []
    for x,y,pen in strokes:
        z = draw_z if pen==1 else lift_z
        samples.append((x,y,z,pen))
    return samples

def compute_kinematics(samples, dt):
    """samples: list of (x,y,z,pen)
       returns arrays t, x,y,z, vx,vy,vz, ax,ay,az"""
    pts = np.array([[s[0], s[1], s[2]] for s in samples], dtype=float)
    n = len(pts)
    if n == 0:
        return None
    # time array
    t = np.arange(n) * dt
    # velocities: forward difference, central for interior
    v = np.zeros_like(pts)
    if n>=2:
        v[0] = (pts[1]-pts[0]) / dt
        for i in range(1,n-1):
            v[i] = (pts[i+1]-pts[i-1])/(2*dt)
        v[-1] = (pts[-1]-pts[-2]) / dt
    # accelerations
    a = np.zeros_like(pts)
    if n>=3:
        a[0] = (v[1]-v[0]) / dt
        for i in range(1,n-1):
            a[i] = (v[i+1]-v[i-1])/(2*dt)
        a[-1] = (v[-1]-v[-2]) / dt
    # package
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    vx,vy,vz = v[:,0], v[:,1], v[:,2]
    ax,ay,az = a[:,0], a[:,1], a[:,2]
    return t, x,y,z, vx,vy,vz, ax,ay,az

def get_max_speed_accel(vx,vy,vz,ax,ay,az):
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    accel = np.sqrt(ax**2 + ay**2 + az**2)
    return speed.max() if len(speed)>0 else 0.0, accel.max() if len(accel)>0 else 0.0

def scale_time_to_limits(dt, vx,vy,vz,ax,ay,az, vmax, amax):
    """Given current kinematics, compute time scale factor s >= 1 to satisfy vmax and amax:
       If we scale time by s (i.e., new dt = dt * s), velocities scale by 1/s, accelerations by 1/s^2.
       So choose s = max( vmax_current / vmax_allowed, sqrt(amax_current / amax_allowed) )"""
    cur_vmax, cur_amax = get_max_speed_accel(vx,vy,vz,ax,ay,az)
    s_v = (cur_vmax / vmax) if vmax>0 and cur_vmax>vmax else 1.0
    s_a = math.sqrt(cur_amax / amax) if amax>0 and cur_amax>amax else 1.0
    s = max(1.0, s_v, s_a)
    return s

# --------------------------
# Main pipeline
# --------------------------

def pipeline(image_path, out_csv, cfg):
    img = read_image_gray(image_path)
    h,w = img.shape
    edges = get_binary_edges(img)
    contours = find_contours(edges)
    print(f"Found {len(contours)} contours (filtered).")
    if len(contours)==0:
        print("No contours found. Exiting.")
        return

    strokes = build_strokes_from_contours(contours, img_shape=(h,w), cfg=cfg)
    print(f"Built {len(strokes)} stroke points (incl. pen flags).")
    # convert to xyz samples
    base_samples = strokes_to_samples(strokes, draw_z=cfg['DRAW_Z'], lift_z=cfg['LIFT_Z'])
    # insert pen delays:
    samples_with_delay = insert_pen_delay_samples(base_samples, cfg['SAMPLING_TIME'], cfg['PEN_DOWN_DELAY'], cfg['PEN_UP_DELAY'])
    print(f"After inserting pen delays: {len(samples_with_delay)} samples.")

    # compute kinematics
    t,x,y,z,vx,vy,vz,ax,ay,az = compute_kinematics(samples_with_delay, cfg['SAMPLING_TIME'])
    cur_vmax, cur_amax = get_max_speed_accel(vx,vy,vz,ax,ay,az)
    print(f"Initial vmax={cur_vmax:.2f} mm/s, amax={cur_amax:.2f} mm/s^2 (limits: vmax={cfg['MAX_VELOCITY']}, amax={cfg['MAX_ACCEL']})")

    # if exceeding limits, scale dt and recompute (simple global scaling)
    s = scale_time_to_limits(cfg['SAMPLING_TIME'], vx,vy,vz,ax,ay,az, cfg['MAX_VELOCITY'], cfg['MAX_ACCEL'])
    if s > 1.0001:
        print(f"Scaling time by factor {s:.3f} to respect limits (increasing sampling time).")
        new_dt = cfg['SAMPLING_TIME'] * s
        t,x,y,z,vx,vy,vz,ax,ay,az = compute_kinematics(samples_with_delay, new_dt)
    else:
        new_dt = cfg['SAMPLING_TIME']

    # Build DataFrame
    df = pd.DataFrame({
        't': t,
        'x': x, 'y': y, 'z': z,
        'vx': vx, 'vy': vy, 'vz': vz,
        'ax': ax, 'ay': ay, 'az': az
    })
    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV to {out_csv} (n={len(df)})")
    return df

# --------------------------
# Run as script
# --------------------------
if __name__ == "__main__":
    cfg = {
        'DRAWING_WIDTH_MM': DRAWING_WIDTH_MM,
        'ROBOT_ORIGIN_XY_MM': ROBOT_ORIGIN_XY_MM,
        'PAPER_OFFSET_MM': PAPER_OFFSET_MM,
        'DRAW_Z': DRAW_Z,
        'LIFT_Z': LIFT_Z,
        'PEN_DOWN_DELAY': PEN_DOWN_DELAY,
        'PEN_UP_DELAY': PEN_UP_DELAY,
        'SAMPLING_TIME': SAMPLING_TIME,
        'MAX_VELOCITY': MAX_VELOCITY,
        'MAX_ACCEL': MAX_ACCEL,
        'PATH_POINT_SPACING': PATH_POINT_SPACING,
        'FLIP_VERTICAL': FLIP_VERTICAL
    }

    # Ensure image exists
    if not Path(IMAGE_PATH).exists():
        print(f"ERROR: IMAGE_PATH '{IMAGE_PATH}' not found.")
    else:
        df = pipeline(IMAGE_PATH, OUTPUT_CSV, cfg)
