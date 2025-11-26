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
OUTPUT_CSV = "ur5_cubic_spline_trajectory.csv"
UR5_DT = 0.02                 # 8ms (Fixed Time Step)

# --- Dimensions & Mapping ---
CANVAS_WIDTH_M = 0.30          # ความกว้างรูปจริง
WALL_DISTANCE_X = 0.50         # ระยะห่างกำแพง (X)
START_Y = -0.15                # เริ่มต้น Y
START_Z = 0.30                 # เริ่มต้น Z

# --- Pen Depths ---
DEPTH_PEN_DOWN = 0.000         # แตะกำแพง
DEPTH_PEN_UP   = -0.050        # ยกห่าง 5cm

# --- Speed Settings ---
# เรากำหนดเป็น "ความเร็วเฉลี่ยที่ต้องการ" แทน
TARGET_SPEED_DRAW = 0.05       # m/s (วาด)
TARGET_SPEED_TRAVEL = 0.25     # m/s (ย้ายที่)

# --- Image Processing ---
IMG_PROCESS_WIDTH = 500
MIN_CONTOUR_LEN = 15

# --- Via Point Settings ---
VIA_POINT_DIST = 0.005         # 5mm: ระยะห่างระหว่าง Via Point (ถ้าถี่ไปหุ่นจะสั่น)

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
    """
    ลดจำนวนจุด (Via Points) เพื่อลด Noise และให้ Spline ทำงานได้สมูทขึ้น
    เก็บเฉพาะจุดที่ห่างกันเกิน min_dist และจุดหัว-ท้าย
    """
    if len(points) < 2: return points
    
    kept_points = [points[0]]
    last_point = points[0]
    
    for i in range(1, len(points) - 1):
        dist = np.linalg.norm(points[i] - last_point)
        if dist >= min_dist:
            kept_points.append(points[i])
            last_point = points[i]
            
    kept_points.append(points[-1]) # เก็บจุดสุดท้ายเสมอ
    return np.array(kept_points)

def generate_cubic_spline_trajectory(points, target_speed, dt, start_time):
    """
    สร้าง Trajectory ด้วย Cubic Spline ผ่านจุด via points
    - points: Nx3 array (x, y, z)
    - target_speed: ความเร็วเฉลี่ยที่ต้องการ
    - bc_type='clamped': บังคับ v=0 ที่หัวและท้าย (Smooth start/stop)
    """
    # 1. คำนวณระยะทางสะสม (Cumulative Distance) เพื่อประมาณเวลา
    # เราใช้ระยะทางในการ map เวลา เพื่อให้ความเร็วสม่ำเสมอ
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.hstack(([0.0], np.cumsum(dists)))
    total_len = cum_dist[-1]
    
    if total_len < 1e-6:
        return [], start_time
    
    # 2. สร้าง Time Knots (จุดเวลาสำหรับ Via Points)
    # สมมติให้วิ่งด้วยความเร็วคงที่ -> t = s / v
    # แต่จริงๆ Cubic Spline จะทำให้ v เปลี่ยนแปลงเล็กน้อยเพื่อให้ path smooth
    total_duration = total_len / target_speed
    
    # ป้องกันเวลาสั้นเกินไปจนความเร่งระเบิด
    min_duration = 0.5  # อย่างน้อย 0.5 วินาทีต่อ stroke (ปรับได้)
    total_duration = max(total_duration, min_duration)
    
    # Map ระยะทาง -> เวลา (Linear mapping for knots)
    t_knots = (cum_dist / total_len) * total_duration
    
    # 3. สร้าง Cubic Spline (x(t), y(t), z(t))
    # bc_type='clamped' หมายถึงบังคับให้ Derivative แรก (ความเร็ว) เป็น 0 ที่ขอบ
    # ซึ่งช่วยลด Jerk ตอนเริ่มและหยุดได้ดีมาก
    cs_x = CubicSpline(t_knots, points[:, 0], bc_type='clamped')
    cs_y = CubicSpline(t_knots, points[:, 1], bc_type='clamped')
    cs_z = CubicSpline(t_knots, points[:, 2], bc_type='clamped')
    
    # 4. Resample ด้วย Fixed DT
    num_steps = int(np.ceil(total_duration / dt))
    t_eval = np.arange(0, num_steps + 1) * dt
    
    # ตัดส่วนเกิน
    t_eval = t_eval[t_eval <= total_duration]
    
    # Evaluate Position
    x_new = cs_x(t_eval)
    y_new = cs_y(t_eval)
    z_new = cs_z(t_eval)
    
    # Evaluate Velocity (Derivative 1)
    vx_new = cs_x(t_eval, 1)
    vy_new = cs_y(t_eval, 1)
    vz_new = cs_z(t_eval, 1)
    
    # Evaluate Acceleration (Derivative 2)
    ax_new = cs_x(t_eval, 2)
    ay_new = cs_y(t_eval, 2)
    az_new = cs_z(t_eval, 2)
    
    # Pack data
    traj_segment = []
    for i in range(len(t_eval)):
        row = [
            start_time + t_eval[i],
            x_new[i], y_new[i], z_new[i],
            vx_new[i], vy_new[i], vz_new[i],
            ax_new[i], ay_new[i], az_new[i]
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

# --- STEP 2: Scale to Wall Workspace ---
print(f"2. Scaling and Creating Via Points (Dist X={WALL_DISTANCE_X}m)...")
scale_factor = CANVAS_WIDTH_M / img_w

strokes_via_points = []
for cnt in contours:
    if len(cnt) < MIN_CONTOUR_LEN: continue
    pts_px = cnt.reshape(-1, 2)
    
    # Map Pixel -> Meter (Wall Plane: Y, Z)
    y_m = pts_px[:, 0] * scale_factor + START_Y
    z_m = (img_h - pts_px[:, 1]) * scale_factor + START_Z
    
    # สร้าง Dense Points (Nx3)
    # X = WALL_DISTANCE_X + DEPTH_PEN_DOWN (ติดกำแพง)
    x_m = np.full_like(y_m, WALL_DISTANCE_X + DEPTH_PEN_DOWN)
    
    dense_points = np.column_stack((x_m, y_m, z_m))
    
    # *** Downsample to get Via Points ***
    # นี่คือขั้นตอนสำคัญที่ลด Jerk จาก Noise ของภาพ
    via_points = downsample_points(dense_points, VIA_POINT_DIST)
    
    strokes_via_points.append(via_points)

# Sort strokes
if strokes_via_points:
    strokes_via_points.sort(key=lambda s: s[0, 1]) # Sort by Y

# --- STEP 3: Generate Cubic Spline Trajectory ---
print("3. Generating Smooth Cubic Spline Trajectory...")

full_traj_data = []
current_time = 0.0
last_pos = None # (x, y, z)

for i, stroke in enumerate(strokes_via_points):
    
    # --- Phase A: Travel (Pen Retracted) ---
    if last_pos is not None:
        start_pt = last_pos
        end_pt = stroke[0]
        
        # จุดผ่านกลางอากาศ (Via Points for Travel)
        # เพื่อให้ยกปากกาโค้งๆ สวยๆ ไม่ใช่เส้นตรงทื่อๆ
        mid_y = (start_pt[1] + end_pt[1]) / 2
        mid_z = (start_pt[2] + end_pt[2]) / 2
        
        # ยกปากกาออกมา (Retract X)
        retract_x = WALL_DISTANCE_X + DEPTH_PEN_UP
        
        # สร้าง 3 Via Points หลัก: [Start, Mid_Air, End]
        # จุด Start (ที่กำแพง)
        p1 = start_pt 
        # จุด Retract Start (ดึงออก)
        p2 = np.array([retract_x, start_pt[1], start_pt[2]])
        # จุด Mid Air (ลอยอยู่)
        p3 = np.array([retract_x, mid_y, mid_z])
        # จุด Retract End (ไปจ่อจุดใหม่)
        p4 = np.array([retract_x, end_pt[1], end_pt[2]])
        # จุด End (จิ้มลงจุดใหม่)
        p5 = end_pt
        
        travel_via_points = np.vstack((p1, p2, p3, p4, p5))
        
        # สร้าง Trajectory สำหรับ Travel
        segment_data, current_time = generate_cubic_spline_trajectory(
            travel_via_points, TARGET_SPEED_TRAVEL, UR5_DT, current_time
        )
        full_traj_data.extend(segment_data)

    # --- Phase B: Drawing (Pen on Wall) ---
    # stroke คือชุดของ Via Points ที่ downsample มาแล้ว
    segment_data, current_time = generate_cubic_spline_trajectory(
        stroke, TARGET_SPEED_DRAW, UR5_DT, current_time
    )
    full_traj_data.extend(segment_data)
    
    # Update last position
    last_pos = stroke[-1]

# --- STEP 4: Save CSV ---
print("4. Saving CSV...")
cols = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
df = pd.DataFrame(full_traj_data, columns=cols)
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
print(f"✅ Saved to {OUTPUT_CSV}")

# --- STEP 5: Plot Validation (Jerk Check) ---
# คำนวณ Jerk (Diff of Acceleration)
dt_arr = np.diff(df['t'])
dt_arr = np.append(dt_arr, dt_arr[-1]) # Padding
jx = np.gradient(df['ax'], df['t'])
jy = np.gradient(df['ay'], df['t'])
jz = np.gradient(df['az'], df['t'])
jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)

plt.figure(figsize=(12, 10))

# 3D Path
ax = plt.subplot(2, 2, 1, projection='3d')
ax.plot(df['x'], df['y'], df['z'], linewidth=0.5)
ax.set_title(f'Smooth Path (Via Dist={VIA_POINT_DIST}m)')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# View Wall
ax.view_init(elev=20, azim=10)

# Velocity
plt.subplot(2, 2, 2)
v_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
plt.plot(df['t'], v_mag)
plt.title('Velocity Magnitude (Smooth Bell Shapes)')
plt.grid(True)

# Acceleration
plt.subplot(2, 2, 3)
a_mag = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
plt.plot(df['t'], a_mag)
plt.title('Acceleration Magnitude (Continuous)')
plt.grid(True)

# Jerk
plt.subplot(2, 2, 4)
plt.plot(df['t'], jerk_mag)
plt.title('Jerk Magnitude (Should be bounded)')
plt.grid(True)

plt.tight_layout()
plt.show()