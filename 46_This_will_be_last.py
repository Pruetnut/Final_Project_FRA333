import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ======================================================================
# 1. CONFIGURATION
# ======================================================================

DRAW_MODE = 'FLOOR'      # 'WALL' or 'FLOOR'
IMAGE_PATH = "image/FIBO.png"
# OUTPUT_CSV = f"5ur5_Trajectory{DRAW_MODE.lower()}.csv"
OUTPUT_CSV = f"02To_worspace.csv"

# Workspace
CANVAS_WIDTH_M = 0.9     # drawing width in meters
IMG_PROCESS_WIDTH = 600   # resize width pixels
MIN_CONTOUR_LEN = 15
VIA_POINT_DIST = 0.005    # downsample 5mm
SMOOTHING_FACTOR = 0.0002 # spline smoothness

# Safe Heights
SAFE_Z_FLOOR = 0.01
SAFE_X_WALL = 0.05

# Pen offsets
PEN_OFFSET_DOWN = 0.0   #m
PEN_OFFSET_UP = 0.05    #m

# Speed
TARGET_SPEED_DRAW = 0.02    #m/s
TARGET_SPEED_TRAVEL = 0.02  #m/s

# Sampling
UR5_DT = 0.008  #s

# Wall/Floor Offsets
if DRAW_MODE == 'WALL':
    START_POS_H = -0.15
    START_POS_V = 0.50
    PLANE_LEVEL = 0.50
elif DRAW_MODE == 'FLOOR':
    START_POS_H = -0.15+0.20
    START_POS_V = 0.40+0.20
    PLANE_LEVEL = 0.0

# ======================================================================
# 2. FUNCTIONS
# ======================================================================

def process_image_to_edges(image_path, target_width):   #input is image out put is edge image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    blur = cv2.GaussianBlur(resized, (5,5),0)
    bilateral = cv2.bilateralFilter(blur,15,75,75)
    edges = cv2.Canny(bilateral,50,150)
    return edges, new_h, target_width

def downsample_points(points, min_dist):  #จำนวนจุดที่อยู่ห่างจากเพื่อนเล็กน้อย
    if len(points) < 2:
        return points
    kept = [points[0]]
    last = points[0]
    for i in range(1,len(points)-1):
        if np.linalg.norm(points[i]-last) >= min_dist:
            kept.append(points[i])
            last = points[i]
    kept.append(points[-1])
    return np.array(kept)

def generate_linear_segment(points, speed, dt, start_t):
    """Generate trajectory with constant speed along points"""
    traj = []
    t = start_t
    # traj = insert_delay(traj, 0.5, dt)

    for i in range(len(points)-1):
        p0 = points[i]
        p1 = points[i+1]
        dist = np.linalg.norm(p1-p0)    #คำนวนระยะระหว่างจุดสองจุด
        duration = max(dist/speed, dt)
        n_step = int(np.ceil(duration/dt))
        for s in range(n_step):
            u = s/n_step
            pos = p0*(1-u) + p1*u
            vel = (p1-p0)/duration
            acc = np.zeros(3)
            traj.append([t, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], acc[0], acc[1], acc[2]])  # t x y x vx vy vz ax ay az
            t += dt
    # append last point
    pos = points[-1]
    traj.append([t, pos[0], pos[1], pos[2], 0,0,0,0,0,0])
    traj = insert_delay(traj, 0.5, dt)

    return traj, t

def generate_spline_segment(points, speed, dt, start_t, smooth_factor):
    """Generate smooth B-spline segment"""
    if len(points)<4:
        return generate_linear_segment(points, speed, dt, start_t)
    pts_t = points.T.tolist()
    tck, _ = splprep(pts_t,s=smooth_factor,k=3)
    u_fine = np.linspace(0,1,1000)
    xy = np.array(splev(u_fine,tck)).T
    dist = np.sqrt(np.sum(np.diff(xy,axis=0)**2,axis=1))
    total_len = np.sum(dist)
    duration = max(total_len/speed, dt)
    n_step = int(np.ceil(duration/dt))
    t_steps = np.linspace(0,duration,n_step)
    u_eval = t_steps/duration
    pos_eval = np.array(splev(u_eval,tck)).T
    vel_eval = np.gradient(pos_eval, dt, axis=0)
    acc_eval = np.gradient(vel_eval, dt, axis=0)
    traj = []
    for i in range(len(t_steps)):
        traj.append([start_t+t_steps[i], pos_eval[i,0], pos_eval[i,1], pos_eval[i,2],
                     vel_eval[i,0], vel_eval[i,1], vel_eval[i,2],
                     acc_eval[i,0], acc_eval[i,1], acc_eval[i,2]])
    return traj, start_t + duration

def insert_delay(traj, delay, dt):
    """Insert a delay by holding the last pose for specified time"""
    if delay <= 0:
        return traj
    
    last = traj[-1]  # [t, x, y, z, vx, vy, vz, ax, ay, az]
    t_last = last[0]
    pos = last[1:4]

    n_step = int(np.ceil(delay / dt))

    for i in range(n_step):
        t_last += dt
        traj.append([t_last, pos[0], pos[1], pos[2],
                     0,0,0, 0,0,0])
    return traj

# ======================================================================
# 3. MAIN WORKFLOW
# ======================================================================

print("1. Processing Image...")
edges, img_h, img_w = process_image_to_edges(IMAGE_PATH, IMG_PROCESS_WIDTH)
cv2.imwrite("edges_output.png", edges)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Mapping contours to workspace
print("2. Mapping & Downsampling...")
scale_factor = CANVAS_WIDTH_M / img_w
strokes_points = []

for cnt in contours:
    if len(cnt)<MIN_CONTOUR_LEN:
        continue
    pts_px = cnt.reshape(-1,2)
    if DRAW_MODE=='WALL':
        y_rob = pts_px[:,0]*scale_factor + START_POS_H
        z_rob = (img_h - pts_px[:,1])*scale_factor + START_POS_V
        x_rob = np.full_like(y_rob, PLANE_LEVEL+PEN_OFFSET_DOWN)
        dense_pts = np.column_stack((x_rob, y_rob, z_rob))
    elif DRAW_MODE=='FLOOR':
        y_rob = pts_px[:,0]*scale_factor + START_POS_H
        x_rob = START_POS_V - pts_px[:,1]*scale_factor
        z_rob = np.full_like(y_rob, PLANE_LEVEL+PEN_OFFSET_DOWN)
        dense_pts = np.column_stack((x_rob, y_rob, z_rob))
    via_pts = downsample_points(dense_pts,VIA_POINT_DIST)
    strokes_points.append(via_pts)

# Sort by first Y for efficiency
if strokes_points:
    strokes_points.sort(key=lambda s:s[0,1])

# Generate full trajectory
print("3. Generating Trajectory...")
full_traj = []
t_curr = 0.0
last_pos = None
SAFE_X = SAFE_X_WALL
SAFE_Z = SAFE_Z_FLOOR

for stroke in strokes_points:
    # Travel pen-up
    if last_pos is not None:
        start_pt = last_pos.copy()
        end_pt = stroke[0].copy()
        travel_pts = np.array([start_pt])
        if DRAW_MODE=='WALL':
            travel_pts = np.array([
                start_pt,
                [SAFE_X, start_pt[1], start_pt[2]],
                [SAFE_X, (start_pt[1]+end_pt[1])/2, (start_pt[2]+end_pt[2])/2],
                [SAFE_X, end_pt[1], end_pt[2]],
                end_pt
            ])
        elif DRAW_MODE=='FLOOR':
            travel_pts = np.array([
                start_pt,
                [start_pt[0], start_pt[1], SAFE_Z],
                [(start_pt[0]+end_pt[0])/2, (start_pt[1]+end_pt[1])/2, SAFE_Z],
                [end_pt[0], end_pt[1], SAFE_Z],
                end_pt
            ])
        seg, t_curr = generate_linear_segment(travel_pts, TARGET_SPEED_TRAVEL, UR5_DT, t_curr)
        full_traj.extend(seg)
    # Draw pen-down
    seg, t_curr = generate_spline_segment(stroke, TARGET_SPEED_DRAW, UR5_DT, t_curr, SMOOTHING_FACTOR)
    full_traj.extend(seg)
    last_pos = stroke[-1]


# Export CSV
print("4. Saving CSV...")
cols = ['t','x','y','z','vx','vy','vz','ax','ay','az']
df = pd.DataFrame(full_traj, columns=cols)

from scipy.signal import savgol_filter

# df: columns t,x,y,z
# 1) ทำ resample ให้ dt สม่ำเสมอ (ใช้ UR5_DT)
dt = UR5_DT  # เช่น 0.008
t_new = np.arange(df['t'].iloc[0], df['t'].iloc[-1], dt)
# สร้าง interpolation ของตำแหน่ง
from scipy.interpolate import interp1d
fx = interp1d(df['t'], df['x'], kind='linear')
fy = interp1d(df['t'], df['y'], kind='linear')
fz = interp1d(df['t'], df['z'], kind='linear')
x_new = fx(t_new); y_new = fy(t_new); z_new = fz(t_new)

# 2) Smooth ตำแหน่งด้วย Savitzky-Golay
# window_length ต้องเป็นเลขคี่ และ <= len(t_new)
win = 11 if len(t_new) >= 11 else (len(t_new)//2)*2+1
poly = 3
x_s = savgol_filter(x_new, win, poly)
y_s = savgol_filter(y_new, win, poly)
z_s = savgol_filter(z_new, win, poly)

# 3) คำนวณ vel & acc (central difference)
vx = np.gradient(x_s, dt)
vy = np.gradient(y_s, dt)
vz = np.gradient(z_s, dt)
ax = np.gradient(vx, dt)
ay = np.gradient(vy, dt)
az = np.gradient(vz, dt)



df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
print(f"✅ Saved: {OUTPUT_CSV}")

# Plot for validation
plt.figure(figsize=(10,8))
v_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
v_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

plt.subplot(2,1,1)
plt.plot(df['t'], v_mag)
plt.title("Velocity Profile")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid(True)

ax = plt.subplot(2,1,2, projection='3d')
ax.plot(df['x'], df['y'], df['z'], linewidth=0.5)
ax.set_title("Trajectory")
plt.tight_layout()
plt.show()