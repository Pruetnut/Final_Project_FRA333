import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. CONFIGURATION (ตั้งค่าที่นี่)
# ==============================================================================

# --- A. โหมดการวาด ---
DRAW_MODE = 'FLOOR'  # 'WALL' หรือ 'FLOOR'

IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = f"ur5_smoothV_bspline_{DRAW_MODE.lower()}.csv"
UR5_DT = 0.008                 # Sampling Time (8ms)

# --- B. Workspace ---
CANVAS_WIDTH_M = 0.30          # ความกว้างรูปจริง

# --- C. พิกัดและทิศทาง (Offset) ---
if DRAW_MODE == 'WALL':
    # Wall Mode (YZ Plane, Depth X)
    START_POS_H = -0.15        # Robot Y
    START_POS_V = 0.50         # Robot Z
    PLANE_LEVEL = 0.50         # Wall Distance X
    
    # ปากกา (แกน X)
    PEN_OFFSET_DOWN = 0.000    # จิ้ม
    PEN_OFFSET_UP   = -0.050   # ถอยหลัง 5cm
    
    # Orientation (หัวชี้ไปทาง +X)
    FIXED_ORIENT = [0.0, np.pi/2, 0.0]

elif DRAW_MODE == 'FLOOR':
    # Floor Mode (XY Plane, Depth Z)
    START_POS_H = -0.15        # Robot Y
    START_POS_V = 0.40         # Robot X
    PLANE_LEVEL = 0.00         # Table Height Z
    
    # ปากกา (แกน Z)
    PEN_OFFSET_DOWN = 0.000    # จิ้ม
    PEN_OFFSET_UP   = 0.050    # ยกขึ้น 5cm
    
    # Orientation (หัวชี้ไปทาง -Z)
    FIXED_ORIENT = [0.0, np.pi, 0.0]

# --- D. ความเร็ว (Speed) ---
TARGET_SPEED_DRAW = 0.05       # m/s (เดินนิ่งๆ)
TARGET_SPEED_TRAVEL = 0.05     # m/s (ย้ายที่เร็วๆ)

# --- E. Smoothing Settings (สำคัญมาก!) ---
IMG_PROCESS_WIDTH = 500
MIN_CONTOUR_LEN = 15
VIA_POINT_DIST = 0.005         # 5mm sampling

# *** NEW: ตัวแปรควบคุมความเนียน ***
# ค่า s ยิ่งมาก = เส้นยิ่งเรียบ (แต่จะเพี้ยนจากรูปเดิมนิดหน่อย)
# ค่า s=0 คือบังคับผ่านทุกจุด (จะเกิดหนาม)
# แนะนำ: 0.0001 ถึง 0.001
SMOOTHING_FACTOR = 0.0005      

# ==============================================================================
# 2. CORE FUNCTIONS (B-Spline Upgraded)
# ==============================================================================

def process_image_to_edges(image_path, target_width):
    img = cv2.imread(image_path, 0)
    if img is None: raise FileNotFoundError(f"Image not found at {image_path}")
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    
    # Pre-processing เพื่อลด Noise ก่อนเข้า B-Spline
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    return edges, new_h, target_width

def downsample_points(points, min_dist):
    if len(points) < 2: return points
    kept = [points[0]]
    last = points[0]
    for i in range(1, len(points) - 1):
        if np.linalg.norm(points[i] - last) >= min_dist:
            kept.append(points[i])
            last = points[i]
    kept.append(points[-1])
    return np.array(kept)

def generate_smooth_bspline_segment(points, speed, dt, start_t, orient, smooth_factor):
    """
    สร้าง Trajectory ด้วย B-Spline Approximation (scipy.interpolate.splprep)
    ช่วยลด Jerk และ Spiky Velocity ได้ดีที่สุด
    """
    # ต้องมีจุดอย่างน้อย 4 จุดเพื่อสร้าง Cubic B-Spline (k=3)
    # ถ้าจุดน้อยกว่านั้น เราจะ Linear Interpolate เอา (เพราะเส้นสั้นมาก ไม่ต้อง smooth)
    if len(points) < 4:
        # Fallback to Linear for very short lines
        total_dist = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        duration = max(total_dist / speed, 0.1)
        num_steps = int(np.ceil(duration / dt))
        t_eval = np.linspace(0, 1, num_steps+1)
        
        # Linear Interp
        traj_segment = []
        for i, u in enumerate(t_eval):
            pos = points[0] + (points[-1] - points[0]) * u
            vel = (points[-1] - points[0]) / duration # Const vel
            acc = np.zeros(3)
            row = [start_t + i*dt, *pos, *orient, *vel, *acc]
            traj_segment.append(row)
        return traj_segment, start_t + duration

    # --- B-Spline Smoothing ---
    # Transpose points to list of coordinates [[x0,x1...], [y0,y1...], [z0,z1...]]
    points_t = points.T.tolist()
    
    # splprep: หา B-Spline representation
    # u: parameter (0 to 1) ตามแนวเส้นโค้ง
    # tck: tuple (knots, coefficients, degree)
    try:
        tck, u = splprep(points_t, s=smooth_factor, k=3) # k=3 (Cubic)
    except Exception as e:
        print(f"Spline Error: {e}, skipping segment.")
        return [], start_t

    # คำนวณความยาวเส้นโค้งใหม่ (Arc Length) เพื่อ map เวลาให้ถูกต้อง
    # เราจะ sample u ละเอียดๆ เพื่อวัดความยาว
    u_fine = np.linspace(0, 1, 1000)
    xy_fine = np.array(splev(u_fine, tck)).T
    dist_fine = np.sqrt(np.sum(np.diff(xy_fine, axis=0)**2, axis=1))
    total_len = np.sum(dist_fine)
    
    # คำนวณเวลา (Duration)
    duration = max(total_len / speed, 0.5)
    
    # สร้าง Time Steps
    num_steps = int(np.ceil(duration / dt))
    t_steps = np.arange(0, num_steps + 1) * dt
    # ตัดส่วนเกิน
    t_steps = t_steps[t_steps <= duration]
    
    # Map Time (t) -> Spline Parameter (u)
    # สมมติความเร็วคงที่: u = t / duration
    u_eval = t_steps / duration
    
    # Evaluate Position (der=0)
    pos_eval = np.array(splev(u_eval, tck, der=0)).T
    
    # Evaluate Velocity (der=1)
    # Note: splev(der=1) gives dx/du. 
    # We need v = dx/dt = (dx/du) * (du/dt)
    # du/dt = 1 / duration
    vel_eval = np.array(splev(u_eval, tck, der=1)).T * (1.0 / duration)
    
    # Evaluate Acceleration (der=2)
    # a = d2x/dt2 = (d2x/du2) * (du/dt)^2
    acc_eval = np.array(splev(u_eval, tck, der=2)).T * ((1.0 / duration)**2)
    
    # บังคับจุดเริ่มต้นและจบให้ความเร็วเป็น 0 (Ramp up/down)
    # (ใช้ Hanning window หรือ sine weight เพื่อเกลี่ยหัวท้ายให้ smooth ขึ้นถ้าต้องการ)
    # แต่ B-Spline ปกติจะ smooth อยู่แล้ว ยกเว้นหัวท้ายที่อาจจะกระโดด
    # เทคนิค: Override 2-3 จุดแรกและท้ายด้วย 0 เพื่อความปลอดภัย
    vel_eval[0] = 0; vel_eval[-1] = 0
    acc_eval[0] = 0; acc_eval[-1] = 0
    
    segment_data = []
    r, p, yaw = orient
    
    for i in range(len(t_steps)):
        row = [
            start_t + t_steps[i],
            pos_eval[i,0], pos_eval[i,1], pos_eval[i,2],  # XYZ
            r, p, yaw,                                    # RPY
            vel_eval[i,0], vel_eval[i,1], vel_eval[i,2],  # V
            acc_eval[i,0], acc_eval[i,1], acc_eval[i,2]   # A
        ]
        segment_data.append(row)
        
    return segment_data, start_t + t_steps[-1]

def generate_linear_segment(p0, p1, speed, dt, start_t, orient):
    

    dist = np.linalg.norm(p1 - p0)
    duration = max(dist / speed, dt)
    steps = max(int(np.ceil(duration/dt)), 1)
    traj = []

    for i in range(steps+1):
        a = i / steps
        pos = (1 - a) * p0 + a * p1
        vel = (p1 - p0) / duration
        acc = np.zeros(3)
        t = start_t + i * dt

        traj.append([
            t, pos[0], pos[1], pos[2],
            orient[0], orient[1], orient[2],
            vel[0], vel[1], vel[2],
            acc[0], acc[1], acc[2]
        ])

    return traj, start_t + duration

def generate_penup_travel(last_end, next_start, dt, start_t, orient,
                          lift_height=0.05,
                          travel_speed=0.10,
                          lift_speed=0.05):

    traj = []

    # (1) Lift Up
    p_lift = last_end.copy()
    p_lift[2] += lift_height
    part, t1 = generate_linear_segment(last_end, p_lift, lift_speed, dt, start_t, orient)
    traj += part

    # (2) Horizontal travel
    p_travel = next_start.copy()
    p_travel[2] = p_lift[2]        # keep same height
    part, t2 = generate_linear_segment(p_lift, p_travel, travel_speed, dt, t1, orient)
    traj += part

    # (3) Lower Down
    p_down = next_start.copy()
    part, t3 = generate_linear_segment(p_travel, p_down, lift_speed, dt, t2, orient)
    traj += part

    return traj, t3


# ==============================================================================
# 3. MAIN WORKFLOW
# ==============================================================================

# --- 1. Process Image ---
print(f"1. Processing Image ({DRAW_MODE})...")
edges, img_h, img_w = process_image_to_edges(IMAGE_PATH, IMG_PROCESS_WIDTH)
cv2.imwrite("egdesformcanny.png",edges)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# --- 2. Map Coordinates & Create Input Points ---
print("2. Mapping & Downsampling...")
scale_factor = CANVAS_WIDTH_M / img_w
strokes_points = []

for cnt in contours:
    if len(cnt) < MIN_CONTOUR_LEN: continue
    pts_px = cnt.reshape(-1, 2)
    
    # Mapping Logic
    if DRAW_MODE == 'WALL':
        y_rob = pts_px[:, 0] * scale_factor + START_POS_H
        z_rob = (img_h - pts_px[:, 1]) * scale_factor + START_POS_V
        x_rob = np.full_like(y_rob, PLANE_LEVEL + PEN_OFFSET_DOWN)
        dense_pts = np.column_stack((x_rob, y_rob, z_rob))
        
    elif DRAW_MODE == 'FLOOR':
        y_rob = pts_px[:, 0] * scale_factor + START_POS_H
        x_rob = START_POS_V - (pts_px[:, 1] * scale_factor)
        z_rob = np.full_like(y_rob, PLANE_LEVEL + PEN_OFFSET_DOWN)
        dense_pts = np.column_stack((x_rob, y_rob, z_rob))

    via_pts = downsample_points(dense_pts, VIA_POINT_DIST)
    strokes_points.append(via_pts)

# Sort optimization
if strokes_points: strokes_points.sort(key=lambda s: s[0, 1])

# --- 3. Generate B-Spline Trajectory ---
print(f"3. Generating B-Spline Path (Smooth factor={SMOOTHING_FACTOR})...")
full_traj_data = []
current_time = 0.0
last_pos = None

# Safety Heights
SAFE_X = PLANE_LEVEL + PEN_OFFSET_UP if DRAW_MODE == 'WALL' else None
SAFE_Z = PLANE_LEVEL + PEN_OFFSET_UP if DRAW_MODE == 'FLOOR' else None

for i, stroke in enumerate(strokes_points):
    

    # --- A. Travel (Lift → Move → Lower) ---
    if last_pos is not None:
        penup_traj, current_time = generate_penup_travel(
            last_pos,          # จุดสิ้นสุดของ stroke ก่อนหน้า
            stroke[0],         # จุดเริ่มต้นของ stroke ถัดไป
            dt=UR5_DT,
            start_t=current_time,
            orient=FIXED_ORIENT,
            lift_height=0.04,       # ปรับได้
            travel_speed=TARGET_SPEED_TRAVEL,
            lift_speed=TARGET_SPEED_LIFT
    )

    full_traj_data.extend(penup_traj)


    # --- B. Drawing ---
    seg_data, current_time = generate_linear_segment(
        stroke, TARGET_SPEED_DRAW, UR5_DT, current_time, FIXED_ORIENT, smooth_factor=SMOOTHING_FACTOR
    )
    full_traj_data.extend(seg_data)
    
    last_pos = stroke[-1]

# --- 4. Export ---
print("4. Saving CSV...")
cols = ['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
df = pd.DataFrame(full_traj_data, columns=cols)
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
print(f"✅ Saved: {OUTPUT_CSV}")

# --- 5. Validation Plots ---
plt.figure(figsize=(10, 8))

# Velocity Profile Check
v_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
plt.subplot(2,1,1)
plt.plot(df['t'], v_mag, label='Speed')
plt.title("Velocity Profile (Should be smooth bells, no spikes)")
plt.xlabel("Time (s)"); plt.ylabel("Speed (m/s)")
plt.grid(True)

# 3D Path
ax = plt.subplot(2,1,2, projection='3d')
ax.plot(df['x'], df['y'], df['z'], linewidth=0.5)
ax.set_title(f"Smooth Trajectory ({DRAW_MODE})")
if DRAW_MODE == 'WALL': ax.view_init(elev=20, azim=10)
else: ax.view_init(elev=45, azim=-90)

plt.tight_layout()
plt.show()