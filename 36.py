import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate # ต้องใช้สำหรับทำ Trajectory

# -------------------- 0. PARAMETERS (ตั้งค่าหุ่นยนต์ตรงนี้) --------------------
# Robot & Physics
UR5_DT = 0.008          # Time Step (8ms สำหรับ UR5 CB3, 2ms สำหรับ e-Series)
V_MAX = 0.15            # m/s (ความเร็วสูงสุด)
A_MAX = 0.3             # m/s^2 (ความเร่งสูงสุด)
SPEED_FACTOR_DRAW = 0.6 # ตอนวาดให้เดินช้าลงเหลือ 60%

# Dimensions (Meters)
# จุดเริ่มวาด (Offset) บนโต๊ะหุ่นยนต์ (สมมติว่ากระดาษวางห่างจากฐาน 40cm, เยื้องซ้าย 20cm)
OFFSET_X = 0.40         
OFFSET_Y = 0.00         

# Pen Heights
Z_UP = 0.05             # ยกปากกาสูง 5cm
Z_DOWN = 0.00           # ปากกาแตะกระดาษ (0cm)

# -------------------- 1. Image Processing (Original Code) --------------------
input_image = "image/FIBO.png" 
# input_image = "image/Temple.jpg"

def process_image_to_edges(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return None
        
    original_height, original_width = img.shape[:2]
    new_width = 600
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    
    g_blurred = cv2.GaussianBlur(resized_image, (5,5), 0)
    bilateral = cv2.bilateralFilter(g_blurred, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    return edges

# Run Step 1
edge_image = process_image_to_edges(input_image)

if edge_image is not None:
    # Scale Calculation
    canvas_width_m = 0.20   # สมมติกระดาษกว้าง 20cm (ปรับตามจริง)
    canvas_height_m = 0.20  # สมมติกระดาษสูง 20cm (ปรับตามจริง)
    
    img_height, img_width = edge_image.shape[:2]
    scale_x = canvas_width_m / img_width
    scale_y = canvas_height_m / img_height

    # -------------------- 2. Contour Extraction (Original Code) --------------------
    def extract_contours_as_paths(edge_image, min_area=100):
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_paths = []
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area or cv2.arcLength(cnt, False) > 50:
                epsilon = 0.001 * cv2.arcLength(cnt, False)
                approx = cv2.approxPolyDP(cnt, epsilon, False)
                valid_paths.append(approx.reshape(-1, 2))
        return valid_paths

    pixel_paths = extract_contours_as_paths(edge_image)
    print(f"Step 2 Done: Found {len(pixel_paths)} contours (pixels).")

    # -------------------- 3. Coordinate Transform & Sorting (NEW) --------------------
    # แปลง Pixel -> Meter และเรียงลำดับเส้น
    
    robot_strokes = []
    
    # 3.1 Transform
    for path in pixel_paths:
        stroke_m = []
        for point in path:
            u, v = point
            # แปลงหน่วย + ใส่ Offset
            x_m = (u * scale_x) + OFFSET_X
            # Flip Y: กลับหัวแกน Y (เพราะในรูป Y ลงล่าง แต่ในโลกจริง Y มักขึ้นบนหรือไปทางซ้าย)
            y_m = ((img_height - v) * scale_y) + OFFSET_Y 
            stroke_m.append([x_m, y_m])
        robot_strokes.append(np.array(stroke_m))

    # 3.2 Sorting (Nearest Neighbor)
    if len(robot_strokes) > 0:
        sorted_strokes = []
        unvisited = robot_strokes.copy()
        current_pos = np.array([OFFSET_X, OFFSET_Y]) # เริ่มที่ Home

        while unvisited:
            # หาเส้นที่จุดเริ่มต้นใกล้หัวปากกาที่สุด
            dists = [np.linalg.norm(s[0] - current_pos) for s in unvisited]
            nearest_idx = np.argmin(dists)
            
            next_stroke = unvisited.pop(nearest_idx)
            sorted_strokes.append(next_stroke)
            current_pos = next_stroke[-1]
        
        robot_strokes = sorted_strokes
        print(f"Step 3 Done: Sorted {len(robot_strokes)} strokes.")

    # -------------------- 4. Trajectory Generation Engine (NEW) --------------------
    # ฟังก์ชันคำนวณ Trapezoidal Velocity Profile
    def generate_s_profile(dist, vmax, amax, dt):
        if dist < 1e-6: return np.array([0.0]), np.array([0.0])
        
        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc**2
        
        if dist < 2 * d_acc: # Triangular
            t_acc = np.sqrt(dist / amax)
            t_flat = 0
            v_peak = amax * t_acc
        else: # Trapezoidal
            d_flat = dist - 2 * d_acc
            t_flat = d_flat / vmax
            v_peak = vmax
            t_acc = t_acc
            
        total_time = 2*t_acc + t_flat
        num_steps = int(np.ceil(total_time / dt))
        t = np.arange(0, num_steps + 1) * dt
        t = t[t <= total_time]
        
        s = np.zeros_like(t)
        for i, time in enumerate(t):
            if time <= t_acc:
                s[i] = 0.5 * amax * time**2
            elif time <= t_acc + t_flat:
                s[i] = d_acc + v_peak * (time - t_acc)
            else:
                t_dec = time - (t_acc + t_flat)
                s[i] = d_acc + (v_peak * t_flat) + (v_peak * t_dec) - (0.5 * amax * t_dec**2)
        
        return np.clip(s, 0, dist), t

    def generate_linear_move(p_start, p_end, vmax, amax, dt):
        dist = np.linalg.norm(p_end - p_start)
        s, t = generate_s_profile(dist, vmax, amax, dt)
        if len(s) == 0: return [], []
        direction = (p_end - p_start) / dist
        pts = p_start + (direction * s[:, np.newaxis])
        return pts, t

    # --- Main Generator Loop ---
    traj_data = [] # [t, x, y, z]
    current_global_time = 0.0

    def append_data(points, t_duration):
        nonlocal current_global_time
        if len(points) == 0: return
        t_seq = current_global_time + t_duration + UR5_DT
        for i in range(len(points)):
            traj_data.append([t_seq[i], points[i][0], points[i][1], points[i][2]])
        if len(t_seq) > 0: current_global_time = t_seq[-1]

    # เริ่มสร้างจุด
    print("Step 4: Generating Trajectory...")
    
    # 1. ยกปากกาเริ่มต้น (Safety)
    if len(robot_strokes) > 0:
        start_x, start_y = robot_strokes[0][0]
        # Warp ไปที่เหนือจุดแรก (ใน simulation ทำได้ แต่ของจริงควรเดินไป)
        # นี่คือจุดเริ่มต้น
        
    for i, stroke in enumerate(robot_strokes):
        p_start_xy = stroke[0]
        p_end_xy = stroke[-1]
        
        # A. APPROACH (ลงปากกา)
        p_up = np.array([p_start_xy[0], p_start_xy[1], Z_UP])
        p_down = np.array([p_start_xy[0], p_start_xy[1], Z_DOWN])
        
        # เดินทางจากจุดปัจจุบัน (ถ้ามี) มาที่ p_up ก่อน (Travel) -> อันนี้ละไว้ในฐานที่เข้าใจ
        # เอาแค่จังหวะแทงปากกาลง
        pts, t = generate_linear_move(p_up, p_down, V_MAX*0.5, A_MAX, UR5_DT)
        append_data(pts, t)
        
        # B. DRAW (ลากเส้น)
        # คำนวณระยะทางรวมของเส้นโค้ง (Arc Length)
        diffs = np.diff(stroke, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.hstack(([0.0], np.cumsum(dists)))
        total_len = cum_dist[-1]
        
        if total_len > 1e-6:
            # Interpolation
            fx = interpolate.interp1d(cum_dist, stroke[:,0])
            fy = interpolate.interp1d(cum_dist, stroke[:,1])
            
            # S-Profile สำหรับการวาด
            s_draw, t_draw = generate_s_profile(total_len, V_MAX*SPEED_FACTOR_DRAW, A_MAX*SPEED_FACTOR_DRAW, UR5_DT)
            
            x_new = fx(s_draw)
            y_new = fy(s_draw)
            z_new = np.full_like(x_new, Z_DOWN)
            
            pts_draw = np.column_stack((x_new, y_new, z_new))
            append_data(pts_draw, t_draw)
            
        # C. RETRACT (ยกปากกา)
        p_end_down = np.array([p_end_xy[0], p_end_xy[1], Z_DOWN])
        p_end_up = np.array([p_end_xy[0], p_end_xy[1], Z_UP])
        
        pts, t = generate_linear_move(p_end_down, p_end_up, V_MAX, A_MAX, UR5_DT)
        append_data(pts, t)
        
        # D. TRAVEL (ไปเส้นถัดไป)
        if i < len(robot_strokes) - 1:
            next_start_xy = robot_strokes[i+1][0]
            p_curr_up = p_end_up
            p_next_up = np.array([next_start_xy[0], next_start_xy[1], Z_UP])
            
            pts, t = generate_linear_move(p_curr_up, p_next_up, V_MAX, A_MAX, UR5_DT)
            append_data(pts, t)

    # -------------------- 5. Save & Visualize --------------------
    traj_arr = np.array(traj_data)
    
    # Save CSV
    output_filename = "final_trajectory.csv"
    header = "t,x,y,z"
    np.savetxt(output_filename, traj_arr, delimiter=",", header=header, comments='')
    print(f"Step 5 Done: Saved {len(traj_arr)} points to {output_filename}")
    
    # Plot 3D Check
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_arr[:,1], traj_arr[:,2], traj_arr[:,3], linewidth=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Final Robot Trajectory (3D)')
    plt.show()

else:
    print("Failed to process image.")