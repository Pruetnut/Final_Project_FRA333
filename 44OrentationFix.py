import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. GLOBAL SETTINGS
# ==============================================================================

IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = "ur5_vertical_wall_corrected.csv"
UR5_DT = 0.008

# --- Workspace & Wall Mapping ---
CANVAS_WIDTH_M = 0.30          # ความกว้างรูปจริง

# กำแพงอยู่ด้านหน้าหุ่น ระยะ X = 0.50m
WALL_DISTANCE_X = 0.70
START_Y = -0.15                # เริ่มต้น Y (ซ้าย-ขวา)
START_Z = 0.30                 # เริ่มต้น Z (ความสูง)

# --- Pen Depth Control (แกน X) ---
# +X คือพุ่งออกจากตัวหุ่นเข้าหากำแพง
DEPTH_PEN_DOWN = 0.000         # จิ้มกำแพง (X = 0.50)
DEPTH_PEN_UP   = -0.050        # ถอยหลัง 5cm (X = 0.45)

# --- Orientation (Fixed) ---
# หัวปากกา (Z-tool) ชี้ไปทาง +X-base (เข้ากำแพง) -> Pitch = +90 deg
FIXED_ROLL = 0.0
FIXED_PITCH = np.pi / 2.0
FIXED_YAW = 0.0

# --- Speed Settings (Average Speed) ---
TARGET_SPEED_DRAW = 0.05       # m/s
TARGET_SPEED_TRAVEL = 0.25     # m/s

# --- Spline & Processing Settings ---
IMG_PROCESS_WIDTH = 500
MIN_CONTOUR_LEN = 15
VIA_POINT_DIST = 0.005         # 5mm: ระยะห่าง Via Point (สำคัญมากเพื่อลด Jerk)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def process_image_to_edges(image_path, target_width):
    img = cv2.imread(image_path, 0)
    if img is None: raise FileNotFoundError("Image not found")
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    return edges, new_h, target_width

def downsample_points(points, min_dist):
    """ลดจำนวนจุดให้เหลือเฉพาะ Via Points ที่จำเป็น"""
    if len(points) < 2: return points
    kept_points = [points[0]]
    last_point = points[0]
    for i in range(1, len(points) - 1):
        dist = np.linalg.norm(points[i] - last_point)
        if dist >= min_dist:
            kept_points.append(points[i])
            last_point = points[i]
    kept_points.append(points[-1])
    return np.array(kept_points)

def generate_cubic_spline_trajectory(points, target_speed, dt, start_time):
    """สร้าง Cubic Spline (x,y,z) พร้อมคำนวณ v, a"""
    # 1. Parameterize by Arc Length
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.hstack(([0.0], np.cumsum(dists)))
    total_len = cum_dist[-1]
    
    if total_len < 1e-6: return [], start_time
    
    # 2. Map Time
    total_duration = total_len / target_speed
    total_duration = max(total_duration, 0.5) # Minimum duration constraint
    t_knots = (cum_dist / total_len) * total_duration
    
    # 3. Create Splines (Clamped BC for smooth stop/start)
    cs_x = CubicSpline(t_knots, points[:, 0], bc_type='clamped')
    cs_y = CubicSpline(t_knots, points[:, 1], bc_type='clamped')
    cs_z = CubicSpline(t_knots, points[:, 2], bc_type='clamped')
    
    # 4. Evaluate with Fixed DT
    num_steps = int(np.ceil(total_duration / dt))
    t_eval = np.arange(0, num_steps + 1) * dt
    t_eval = t_eval[t_eval <= total_duration]
    
    traj_segment = []
    for i in range(len(t_eval)):
        t = t_eval[i]
        row = [
            start_time + t,
            cs_x(t), cs_y(t), cs_z(t),             # Pos
            FIXED_ROLL, FIXED_PITCH, FIXED_YAW,    # Orientation
            cs_x(t, 1), cs_y(t, 1), cs_z(t, 1),    # Vel
            cs_x(t, 2), cs_y(t, 2), cs_z(t, 2)     # Accel
        ]
        traj_segment.append(row)
        
    return traj_segment, start_time + t_eval[-1]

# ==============================================================================
# 3. MAIN WORKFLOW
# ==============================================================================

# --- STEP 1: Process Image ---
print("1. Processing Image...")
edges, img_h, img_w = process_image_to_edges(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# --- STEP 2: Scale & Create Via Points (Wall Mode) ---
print(f"2. Scaling to Wall & Generating Via Points...")
scale_factor = CANVAS_WIDTH_M / img_w

strokes_via_points = []
for cnt in contours:
    if len(cnt) < MIN_CONTOUR_LEN: continue
    pts_px = cnt.reshape(-1, 2)
    
    # Mapping: Img X -> Robot Y, Img Y -> Robot Z
    y_m = pts_px[:, 0] * scale_factor + START_Y
    z_m = (img_h - pts_px[:, 1]) * scale_factor + START_Z
    
    # X constant at Wall (Pen Down)
    x_m = np.full_like(y_m, WALL_DISTANCE_X + DEPTH_PEN_DOWN)
    
    # Combine to (N, 3)
    dense_points = np.column_stack((x_m, y_m, z_m))
    
    # Downsample for Spline
    via_points = downsample_points(dense_points, VIA_POINT_DIST)
    strokes_via_points.append(via_points)

if strokes_via_points: strokes_via_points.sort(key=lambda s: s[0,1]) # Sort by Y

# --- STEP 3: Generate Cubic Spline Trajectory ---
print("3. Generating Smooth Spline Trajectory...")

full_traj_data = []
current_time = 0.0
last_pos = None

# Absolute X positions
ABS_X_WALL    = WALL_DISTANCE_X + DEPTH_PEN_DOWN
ABS_X_RETRACT = WALL_DISTANCE_X + DEPTH_PEN_UP

for i, stroke in enumerate(strokes_via_points):
    
    # --- Phase A: Travel (Smooth Retraction) ---
    if last_pos is not None:
        start_pt = last_pos
        end_pt   = stroke[0]
        
        # จุดกึ่งกลาง (Midpoint)
        mid_y = (start_pt[1] + end_pt[1]) / 2
        mid_z = (start_pt[2] + end_pt[2]) / 2
        
        # สร้าง 5 Control Points สำหรับการยกย้ายที่
        # 1. จุดเริ่ม (ที่กำแพง)
        p1 = start_pt
        # 2. ถอยหลังออกมา (Retract Start)
        p2 = np.array([ABS_X_RETRACT, start_pt[1], start_pt[2]])
        # 3. ลอยไปตรงกลาง (Mid Air)
        p3 = np.array([ABS_X_RETRACT, mid_y, mid_z])
        # 4. ไปจ่อจุดใหม่ (Retract End)
        p4 = np.array([ABS_X_RETRACT, end_pt[1], end_pt[2]])
        # 5. จิ้มจุดใหม่ (Target Wall)
        p5 = end_pt
        
        travel_points = np.vstack((p1, p2, p3, p4, p5))
        
        segment_data, current_time = generate_cubic_spline_trajectory(
            travel_points, TARGET_SPEED_TRAVEL, UR5_DT, current_time
        )
        full_traj_data.extend(segment_data)

    # --- Phase B: Drawing (Spline on Wall) ---
    segment_data, current_time = generate_cubic_spline_trajectory(
        stroke, TARGET_SPEED_DRAW, UR5_DT, current_time
    )
    full_traj_data.extend(segment_data)
    
    last_pos = stroke[-1]

# --- STEP 4: Save CSV ---
print("4. Saving CSV...")
cols = ['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
df = pd.DataFrame(full_traj_data, columns=cols)
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
print(f"✅ Saved to {OUTPUT_CSV}")

# --- STEP 5: 3D Plot Check ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['z'], linewidth=0.8, label='Smooth Path')
ax.set_xlabel('X (Depth)'); ax.set_ylabel('Y (Width)'); ax.set_zlabel('Z (Height)')
ax.set_title(f'Wall Plotter (Smooth Cubic Spline)\nRetract X: {ABS_X_RETRACT:.2f} -> Wall X: {ABS_X_WALL:.2f}')

# มุมมองจากด้านข้าง เพื่อเช็คการถอยหลัง (Retraction)
ax.view_init(elev=30, azim=-60)
plt.show()

# Plot Velocity Profile Check
plt.figure(figsize=(10, 4))
v_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
plt.plot(df['t'], v_mag)
plt.title('Velocity Magnitude (Bell-Shaped = Smooth)')
plt.grid(True)
plt.show()