import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# -------------------- PARAMETERS --------------------
image_path = "image/FIBO.png"       
outputname = "trajectory_fixed_dt_ready.csv"

# Robot Control Parameters
UR5_DT = 0.008                     # *** Fixed Sampling Time (8ms for UR5 CB3) ***
                                   # ใช้ 0.002 สำหรับ UR5 e-Series

pixel_size = 0.001                 # meters per pixel
min_contour_len_px = 20            # ตัดเส้นสั้นทิ้ง
z_down = 0.01                      # ระดับปากกาลง
z_up = 0.05                        # ระดับปากกายก

# Kinematic Limits
vmax_tcp = 0.5                     # m/s
amax_tcp = 1.0                     # m/s^2
pen_down_speed_factor = 0.7        # ลดความเร็วตอนวาดเหลือ 70%

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
    """
    สร้าง Profile ระยะทาง s(t) ตามเวลาที่คงที่ (Fixed DT)
    Returns:
        s_values: ระยะทางสะสม ณ เวลา t
        t_values: เวลา t (local time เริ่มจาก 0)
    """
    if L <= 1e-6:
        return np.array([0.0]), np.array([0.0])

    # 1. คำนวณเวลาที่ใช้ในช่วงต่างๆ
    t_acc = vmax / amax        # เวลาเร่ง
    d_acc = 0.5 * amax * t_acc**2 # ระยะทางเร่ง

    if L < 2 * d_acc:
        # Triangular Profile (ระยะสั้น เร่งไม่ถึง vmax)
        t_acc = np.sqrt(L / amax)
        t_flat = 0
        v_peak = amax * t_acc
        total_time = 2 * t_acc
    else:
        # Trapezoidal Profile (ระยะปกติ)
        d_flat = L - 2 * d_acc
        t_flat = d_flat / vmax
        v_peak = vmax
        total_time = 2 * t_acc + t_flat

    # 2. สร้างแกนเวลาแบบ Fixed Step *** จุดสำคัญ ***
    # สร้างเวลาตั้งแต่ 0 ถึง total_time โดยห่างกันทีละ dt
    # ใช้ num = int(total_time / dt) + 1 เพื่อความแม่นยำ
    num_steps = int(np.ceil(total_time / dt))
    t_values = np.arange(0, num_steps + 1) * dt
    
    # ตัดส่วนเกินที่อาจจะเลย total_time ไปนิดหน่อย (ป้องกัน overshoot)
    t_values = t_values[t_values <= total_time]
    # บังคับให้จุดสุดท้ายคือ total_time เป๊ะๆ (เพื่อจบที่ระยะ L พอดี)
    if len(t_values) == 0 or t_values[-1] < total_time:
        t_values = np.append(t_values, total_time)

    # 3. คำนวณ s(t) ตามสูตร Kinematics
    s_values = np.zeros_like(t_values)
    
    for i, t in enumerate(t_values):
        if t <= t_acc:
            # Acceleration
            s_values[i] = 0.5 * amax * t**2
        elif t <= t_acc + t_flat:
            # Constant Velocity
            s_values[i] = d_acc + v_peak * (t - t_acc)
        else:
            # Deceleration
            t_dec = t - (t_acc + t_flat)
            s_values[i] = d_acc + (v_peak * t_flat) + (v_peak * t_dec) - (0.5 * amax * t_dec**2)
            
    # Clamp ค่าสุดท้ายให้ไม่เกิน L (แก้ปัญหา Floating point error)
    s_values = np.clip(s_values, 0, L)
    
    return s_values, t_values

# -------------------- MAIN PROCESS --------------------

# 1. Image Processing
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None: raise FileNotFoundError(f"Cannot open {image_path}")

h, w = img.shape
img_blur = cv2.GaussianBlur(img, (gauss_ksize, gauss_ksize), 0)
edges = cv2.Canny(img_blur, canny_thresh1, canny_thresh2)
contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

# 2. Extract Strokes & Sort
strokes = []
for cnt in contours:
    if len(cnt) < min_contour_len_px: continue
    pts_m = pixel_coords_to_metric(cnt, h, pixel_size)
    strokes.append(pts_m)

# Sort strokes (Heuristic)
strokes = sorted(strokes, key=lambda s: np.mean(s[:,0]) + np.mean(s[:,1]))


    # ... (โค้ดส่วนบนเหมือนเดิม) ...

# 3. Build Trajectory with Fixed DT
traj_data = [] # List to store [t, x, y, z]
current_global_time = 0.0

print(f"Generating trajectory with fixed dt = {UR5_DT}s ...")

for i, stroke in enumerate(strokes):
    # --- A. Pen Down (Drawing) ---
    # คำนวณความยาวเส้น (Arc Length)
    diffs = np.diff(stroke, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    
    # สร้าง Arc-Length Parameterization สำหรับ Interpolate
    cum_dist = np.hstack(([0.0], np.cumsum(seg_lengths)))
    
    # *** แก้ไขจุดที่ 1: ใช้ค่า L จาก cum_dist ตัวสุดท้ายเพื่อให้เท่ากันเป๊ะ ***
    L = cum_dist[-1]
    
    if L > 1e-6:
        # *** แก้ไขจุดที่ 2: เพิ่ม bounds_error=False และ fill_value ***
        # เพื่อป้องกัน Error ถ้า s เกิน L ไปนิดเดียว (Clamping)
        interp_x = interpolate.interp1d(cum_dist, stroke[:,0], kind='linear', 
                                        bounds_error=False, fill_value=(stroke[0,0], stroke[-1,0]))
        interp_y = interpolate.interp1d(cum_dist, stroke[:,1], kind='linear', 
                                        bounds_error=False, fill_value=(stroke[0,1], stroke[-1,1]))
        
        # คำนวณ s profile แบบ Fixed DT
        v_use = vmax_tcp * pen_down_speed_factor
        a_use = amax_tcp * pen_down_speed_factor
        s_prof, t_prof = generate_trapezoidal_s_profile_fixed_dt(L, v_use, a_use, UR5_DT)
        
        # แปลง s เป็น (x, y) และบันทึก
        x_new = interp_x(s_prof)
        y_new = interp_y(s_prof)
        z_new = np.full_like(x_new, z_down)
        
        # เวลา Global ต่อเนื่อง
        t_global = current_global_time + t_prof
        
        # เก็บลง List
        for j in range(len(t_global)):
            traj_data.append([t_global[j], x_new[j], y_new[j], z_new[j]])
            
        current_global_time = t_global[-1]
    
    # --- B. Pen Up (Travel to next stroke) ---
    if i < len(strokes) - 1:
        start_pt = stroke[-1]           # ปลายเส้นปัจจุบัน
        end_pt = strokes[i+1][0]        # เริ่มต้นเส้นถัดไป
        
        dist_travel = np.linalg.norm(end_pt - start_pt)
        
        if dist_travel > 1e-6:
            # สร้าง Profile สำหรับการเดินทาง
            s_prof_up, t_prof_up = generate_trapezoidal_s_profile_fixed_dt(dist_travel, vmax_tcp, amax_tcp, UR5_DT)
            
            # Linear Interpolation ระหว่างจุด
            direction = (end_pt - start_pt) / dist_travel
            
            x_up = start_pt[0] + direction[0] * s_prof_up
            y_up = start_pt[1] + direction[1] * s_prof_up
            z_up_arr = np.full_like(x_up, z_up)
            
            # เวลา Global ต่อเนื่อง (Shift เวลา Local ให้ต่อจาก Global)
            # หมายเหตุ: เริ่มจุดถัดไปที่ dt ถัดไปเพื่อไม่ให้เวลาซ้ำ
            t_global_up = current_global_time + t_prof_up + UR5_DT 
            
            for j in range(len(t_global_up)):
                traj_data.append([t_global_up[j], x_up[j], y_up[j], z_up_arr[j]])
                
            current_global_time = t_global_up[-1]

# ... (โค้ดส่วนล่างเหมือนเดิม) ...

# -------------------- FINALIZE & SAVE --------------------
traj_arr = np.array(traj_data)

# คำนวณ Velocity & Acceleration (Finite Difference)
# เพราะ dt คงที่ การคำนวณจึงง่ายมาก
vel = np.zeros_like(traj_arr[:, 1:4])
acc = np.zeros_like(traj_arr[:, 1:4])

# v[i] = (p[i] - p[i-1]) / dt
vel[1:] = (traj_arr[1:, 1:4] - traj_arr[:-1, 1:4]) / UR5_DT
# a[i] = (v[i] - v[i-1]) / dt
acc[1:] = (vel[1:] - vel[:-1]) / UR5_DT

# Compute norms
v_norm = np.linalg.norm(vel, axis=1)
a_norm = np.linalg.norm(acc, axis=1)

# Combine for CSV
final_output = np.column_stack((traj_arr, v_norm, a_norm))
header = "t,x,y,z,v,accel"

np.savetxt(outputname, final_output, delimiter=",", header=header, comments='')

print(f"Saved {outputname}")
print(f"Total Points: {len(final_output)}")
print(f"Total Time: {final_output[-1, 0]:.2f} s")
print(f"Fixed Time Step: {UR5_DT} s")

# -------------------- PLOT --------------------
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(traj_arr[:, 1], traj_arr[:, 2], '.-', markersize=1, linewidth=0.5)
plt.title(f"Trajectory Path (Fixed dt={UR5_DT}s)")
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.subplot(2, 1, 2)
plt.plot(traj_arr[:, 0], v_norm, label='Velocity')
plt.title("Velocity Profile")
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.grid(True)

plt.tight_layout()
plt.show()