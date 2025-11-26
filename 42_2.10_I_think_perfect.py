import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. GLOBAL SETTINGS (VERTICAL PLOTTER MODE)
# ==============================================================================

IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = "ur5_vertical_plotter.csv"
UR5_DT = 0.008

# --- Workspace Settings (Physical Dimensions) ---
CANVAS_WIDTH_M = 0.30          # ความกว้างรูปจริง (เมตร)

# *** พิกัดกำแพง (Wall Coordinates) ***
# สมมติกำแพงอยู่ด้านหน้าหุ่นยนต์ (Front Plane)
WALL_DISTANCE_X = 0.50         # กำแพงอยู่ห่างจากฐานไปข้างหน้า 50 ซม. (แกน X)
START_Y = -0.15                # จุดเริ่มวาด (ซ้าย-ขวา) เทียบกับฐาน (แกน Y)
START_Z = 0.30                 # จุดเริ่มวาด (ความสูงจากพื้น) (แกน Z)

# *** การควบคุมปากกา (ควบคุมด้วยแกน X) ***
# X มาก = ยื่นไปข้างหน้า, X น้อย = ดึงกลับหาตัว
DEPTH_PEN_DOWN = 0.000         # ปากกาแตะกำแพงพอดี (Offset จาก WALL_DISTANCE_X เป็น 0)
DEPTH_PEN_UP   = -0.050        # ถอยปากกาออกมา 5 ซม. (เข้าหาตัวหุ่น)

# --- Dynamic Profiles ---
V_DRAW = 0.05
A_DRAW = 0.10
V_TRAVEL = 0.30
A_TRAVEL = 0.50

# --- Image Processing ---
IMG_PROCESS_WIDTH = 500
MIN_CONTOUR_LEN = 15

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def process_image_to_edges(image_path, target_width):
    img = cv2.imread(image_path, 0)
    if img is None: raise FileNotFoundError(f"Image not found")
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    return edges, new_h, target_width

def generate_trapezoidal_profile(dist, v_max, a_max, dt):
    if dist < 1e-6: return np.array([0.0]), np.array([0.0]), np.array([0.0])
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if dist < 2 * d_acc:
        t_acc = np.sqrt(dist / a_max); t_flat = 0; v_peak = a_max * t_acc
    else:
        d_flat = dist - 2 * d_acc; t_flat = d_flat / v_max; v_peak = v_max
    total_time = 2 * t_acc + t_flat
    num_steps = int(np.ceil(total_time / dt))
    t_arr = np.arange(0, num_steps + 1) * dt
    s_arr = np.zeros_like(t_arr); v_arr = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t <= t_acc: s_arr[i] = 0.5*a_max*t**2; v_arr[i] = a_max*t
        elif t <= t_acc+t_flat: s_arr[i] = d_acc + v_peak*(t-t_acc); v_arr[i] = v_peak
        elif t <= total_time:
            t_dec = t-(t_acc+t_flat)
            s_arr[i] = d_acc + v_peak*t_flat + v_peak*t_dec - 0.5*a_max*t_dec**2
            v_arr[i] = v_peak - a_max*t_dec
        else: s_arr[i] = dist; v_arr[i] = 0
    s_arr = np.clip(s_arr, 0, dist)
    return s_arr, v_arr, t_arr

# ==============================================================================
# 3. MAIN WORKFLOW
# ==============================================================================

# --- STEP 1: Process Image ---
print("1. Processing Image...")
edges, img_h, img_w = process_image_to_edges(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# --- STEP 2: Scale to Wall Workspace ---
print(f"2. Scaling to Wall (Dist X={WALL_DISTANCE_X}m)...")
scale_factor = CANVAS_WIDTH_M / img_w

scaled_strokes = []
for cnt in contours:
    if len(cnt) < MIN_CONTOUR_LEN: continue
    pts_px = cnt.reshape(-1, 2)
    
    # *** KEY CHANGE: MAPPING ***
    # Image X (Width) -> Robot Y
    y_m = pts_px[:, 0] * scale_factor + START_Y
    
    # Image Y (Height) -> Robot Z (Invert because img Y is down)
    z_m = (img_h - pts_px[:, 1]) * scale_factor + START_Z
    
    # เก็บเป็น (Y, Z) แทน (X, Y) เพื่อความไม่งง
    scaled_strokes.append(np.column_stack((y_m, z_m)))

if scaled_strokes: scaled_strokes.sort(key=lambda s: s[0,0])

# --- STEP 3: Generate Trajectory ---
print("3. Generating Vertical Trajectory...")
full_traj_data = []
current_time = 0.0
last_yz = None # เก็บตำแหน่งล่าสุด (Y, Z)

# ตำแหน่ง X จริง (Depth)
ABS_X_WALL = WALL_DISTANCE_X + DEPTH_PEN_DOWN
ABS_X_RETRACT = WALL_DISTANCE_X + DEPTH_PEN_UP

for i, stroke in enumerate(scaled_strokes): # stroke ตอนนี้คือ [Y, Z]
    
    # --- Phase A: Travel (Pen Retracted) ---
    if last_yz is not None:
        # ย้ายจากจุดเดิม (ถอยหลัง) -> ไปจุดใหม่ (ถอยหลัง)
        # Start: (X_RETRACT, Last_Y, Last_Z)
        travel_start = np.array([ABS_X_RETRACT, last_yz[0], last_yz[1]])
        # End:   (X_RETRACT, New_Y, New_Z)
        travel_end   = np.array([ABS_X_RETRACT, stroke[0,0], stroke[0,1]])
        
        dist = np.linalg.norm(travel_end - travel_start)
        if dist > 1e-6:
            s_prof, v_prof_scalar, t_prof = generate_trapezoidal_profile(dist, V_TRAVEL, A_TRAVEL, UR5_DT)
            direction = (travel_end - travel_start) / dist
            
            for k in range(len(t_prof)):
                pos = travel_start + direction * s_prof[k]
                vel = direction * v_prof_scalar[k]
                full_traj_data.append([current_time + t_prof[k], pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
            current_time += t_prof[-1]

        # (Optional) Push Pen Forward (X_RETRACT -> X_WALL)
        # เพื่อความสมูท อาจจะถือว่ามันจิ้มเร็วมาก หรือเพิ่ม Trajectory ช่วงจิ้มตรงนี้ได้

    # --- Phase B: Drawing (Pen on Wall) ---
    diffs = np.diff(stroke, axis=0)
    seg_dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.hstack(([0.0], np.cumsum(seg_dists)))
    total_len = cum_dist[-1]
    
    if total_len > 1e-6:
        # Interpolate Y, Z ตามระยะทาง
        fy = interpolate.interp1d(cum_dist, stroke[:,0], kind='linear')
        fz = interpolate.interp1d(cum_dist, stroke[:,1], kind='linear')
        
        s_prof, v_prof_scalar, t_prof = generate_trapezoidal_profile(total_len, V_DRAW, A_DRAW, UR5_DT)
        
        y_draw = fy(s_prof)
        z_draw = fz(s_prof)
        x_draw = np.full_like(y_draw, ABS_X_WALL) # X คงที่ (ติดกำแพง)
        
        for k in range(len(t_prof)):
            # Tangent vector อยู่ในระนาบ YZ
            s_curr = s_prof[k]
            s_next = min(s_curr + 1e-4, total_len)
            
            p_curr = np.array([float(fy(s_curr)), float(fz(s_curr))])
            p_next = np.array([float(fy(s_next)), float(fz(s_next))])
            
            vec = p_next - p_curr
            norm = np.linalg.norm(vec)
            tangent_yz = vec / norm if norm > 0 else np.array([0.0, 0.0])
            
            vx = 0.0 # ไม่มีความเร็วแกน X ตอนวาด (ยกเว้น force control)
            vy = tangent_yz[0] * v_prof_scalar[k]
            vz = tangent_yz[1] * v_prof_scalar[k]
            
            full_traj_data.append([current_time + t_prof[k], x_draw[k], y_draw[k], z_draw[k], vx, vy, vz])
            
        current_time += t_prof[-1]
        last_yz = np.array([y_draw[-1], z_draw[-1]])

# --- STEP 4: Save CSV ---
print("4. Saving CSV...")
df = pd.DataFrame(full_traj_data, columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
df['ax'] = np.gradient(df['vx'], df['t'])
df['ay'] = np.gradient(df['vy'], df['t'])
df['az'] = np.gradient(df['vz'], df['t'])
df = df[['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']]
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
print(f"✅ Saved to {OUTPUT_CSV}")

# --- STEP 5: 3D Plot (ปรับมุมมองให้เห็นเป็นกำแพง) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['z'], linewidth=0.8, label='Wall Path')

# ตั้งค่าแกนให้สมจริง
ax.set_xlabel('X (Depth)'); ax.set_ylabel('Y (Width)'); ax.set_zlabel('Z (Height)')
ax.set_title('Vertical Plotter Trajectory (Wall at X={:.2f})'.format(WALL_DISTANCE_X))

# Fix limits to visualize wall orientation
mid_y, mid_z = df['y'].mean(), df['z'].mean()
ax.set_ylim(mid_y - 0.2, mid_y + 0.2)
ax.set_zlim(mid_z - 0.2, mid_z + 0.2)
ax.set_xlim(0, WALL_DISTANCE_X + 0.2) # Show from base to wall

# มุมมอง (View) ให้เหมือนยืนมองกำแพงจากด้านหลังหุ่น
ax.view_init(elev=20, azim=10) 

plt.show()