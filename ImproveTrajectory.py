import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.spatial import distance
from typing import List, Tuple, Dict

#select import image file 
filename = "image/FIBO.png"
# filename = "image/Bird.jpg"

#(px) The size of out put image after Edge detection.
output_Width = 300
#Canny Edge Detection [adjust parameter]
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 300

#CANNY function
def image_edge_detection(filename, new_width, t1, t2):
    #load image
    ori_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    #resize image
    original_hight, original_width = ori_image.shape[:2]
    aspect_ratio = new_width/original_width
    new_hight = int(original_hight * aspect_ratio)
    resized_image = cv2.resize(ori_image, (new_width, new_hight))
    #reduce noise
    blur = cv2.GaussianBlur(resized_image, (7, 7), 1.1)
    
    edge_image = cv2.Canny(blur, t1, t2) # you can change between "smooth" or "blur" to see the difference result
    return edge_image, resized_image.shape[0]
    
#used the image function
try:
    edges, H = image_edge_detection(filename, output_Width, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
except FileNotFoundError as e:
    print(e)
    exit()

#start plotting
y, x = np.nonzero(edges)
coords = np.column_stack((x, y))  # x = column, y = row

# --- 3. Y-Axis Flip (การแปลงจาก Image Coordinate สู่ Plotter Coordinate) ---
# Image Origin: Top-Left (Y เพิ่มลงล่าง)
# Plotter Origin: Bottom-Left/Center (Y เพิ่มขึ้นบน)
coords[:, 1] = H - 1 - coords[:, 1]

# บันทึกพิกัดขอบภาพที่ปรับแล้วลงใน CSV (พร้อมสำหรับ Path Planning)
np.savetxt('edges_xy_flipped.csv', coords, delimiter=',', header='X,Y', comments='')
print(f"บันทึกพิกัดขอบภาพที่ปรับปรุงแล้วลงใน 'edges_xy_flipped.csv' (จำนวน {len(coords)} จุด)")

#Show result1 : plot Edge detection = coordinate X Y
plt.figure(figsize=(10, 5))

# Plot 1: ขอบภาพ (Edges)
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title(f"Canny Edges (T1={CANNY_THRESHOLD1}, T2={CANNY_THRESHOLD2})")
plt.axis('off')

# Plot 2: พิกัดขอบภาพ (Plotter View)
plt.subplot(1, 2, 2)
# ใช้ scatter plot เพื่อแสดงตำแหน่งของจุดขอบ
plt.scatter(coords[:, 0], coords[:, 1], s=1, color='blue')
# ตั้งค่าแกนเพื่อให้เห็นภาพในมุมมองของ Plotter (Y ขึ้นบน)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"Plotter Coordinates (Flipped Y)")
plt.xlabel("X (Pixel Unit)")
plt.ylabel("Y (Pixel Unit)")

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------
#Path planing make order for all coordinate
W = 300 #[Px]
H = 200 #[Px]

# การแมปพิกัดภาพ (Pixel) ไปยังพิกัดหุ่นยนต์ (Meter)
DRAWING_SIZE_M = 0.200 # ขนาดภาพที่วาดจริง 200 mm
SCALING_FACTOR = DRAWING_SIZE_M / W
X_OFFSET = 0.400 
Y_OFFSET = -0.200

# ความสูงของปากกาและทิศทางของหัวจับ
Z_DRAW = 0.010  # Pen Down (m)
Z_LIFT = 0.050  # Pen Up (m)
ROLL = 0.0
PITCH = -np.pi / 2.0  # -90 องศา (ตั้งฉากกับพื้นผิว)
YAW = 0.0

# Threshold: ระยะห่างสูงสุดในการเชื่อมจุด (Pixels)[adjust parameter]
MAX_GAP_PIXELS = 3

#Segment Generation (Segmented Greedy Nearest Neighbor Search) make path in segment of drawing
def organize_and_segment_path(coords: np.ndarray, max_gap: int) -> List[List[Tuple[float, float]]]:
    points_remaining = set(map(tuple, coords.tolist()))
    path_segments: List[List[Tuple[float, float]]] = []

    while points_remaining:
        start_point = points_remaining.pop()
        current_segment = [start_point]
        current_point = start_point

        while True:
            remaining_array = np.array(list(points_remaining))
            if len(remaining_array) == 0:
                break
            
            dists_sq = np.sum((remaining_array - current_point)**2, axis=1)
            nearest_index = np.argmin(dists_sq)
            min_dist_sq_found = dists_sq[nearest_index]
            
            if min_dist_sq_found <= max_gap**2:
                best_neighbor = tuple(remaining_array[nearest_index])
                points_remaining.remove(best_neighbor)
                current_segment.append(best_neighbor)
                current_point = best_neighbor
            else:
                break 
        
        path_segments.append(current_segment)
        
    return path_segments

#Greedy Traveling Salesperson Problem (TSP) เพื่อหาลำดับการวาดเส้นที่สั้นที่สุด
def optimize_segment_order(segments: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    """เรียงลำดับ segments ใหม่ด้วย Greedy TSP เพื่อลดระยะยกปากกา"""
    optimized_order = []
    if not segments: return optimized_order
    
    remaining_segments = list(segments)
    current_segment = remaining_segments.pop(0)
    optimized_order.append(current_segment)
    
    while remaining_segments:
        current_end = current_segment[-1]
        min_distance = float('inf')
        best_next_index = -1
        
        # ค้นหา segment ถัดไปที่ใกล้ที่สุด
        for i, next_segment in enumerate(remaining_segments):
            next_start = next_segment[0]
            dist = distance.euclidean(current_end, next_start)
            
            if dist < min_distance:
                min_distance = dist
                best_next_index = i
        
        if best_next_index != -1:
            current_segment = remaining_segments.pop(best_next_index)
            optimized_order.append(current_segment)
        else:
            break
            
    return optimized_order

# Trajectory Generation (สร้างไฟล์คำสั่งสำหรับ UR5)
# -----------------------------------------------------------------

def generate_ur5_trajectory(path_segments: List[List[Tuple[float, float]]]) -> pd.DataFrame:
    """แปลง segments ให้เป็นพิกัด Cartesian และเพิ่ม Z/RPY/Speed Profile"""
    final_ur5_trajectory: List[Dict] = []
    
    # ถ้ามีหลาย segment ให้เรียงลำดับก่อน
    path_segments = optimize_segment_order(path_segments)

    for segment in path_segments:
        # A. เตรียมการเคลื่อนที่ (Rapid Traverse)
        start_x_pixel, start_y_pixel = segment[0]
        
        # แปลงพิกัดภาพ (Pixel) เป็นพิกัดหุ่นยนต์ (Meter)
        X_start_m = start_x_pixel * SCALING_FACTOR + X_OFFSET
        Y_start_m = start_y_pixel * SCALING_FACTOR + Y_OFFSET
        
        # คำสั่ง 1: Move_Lift (Pen Up) ไปยังจุดเริ่มต้นของ segment
        final_ur5_trajectory.append({
            'X': X_start_m, 'Y': Y_start_m, 'Z': Z_LIFT, 
            'Roll': ROLL, 'Pitch': PITCH, 'Yaw': YAW, 
            'Action': 'Move_Lift', 'Speed_Profile': 'Rapid'
        })
        
        # คำสั่ง 2: ลงปากกา (Pen Down) ณ จุดเริ่มต้น (รักษาพิกัด X, Y เดิม)
        final_ur5_trajectory.append({
            'X': X_start_m, 'Y': Y_start_m, 'Z': Z_DRAW, 
            'Roll': ROLL, 'Pitch': PITCH, 'Yaw': YAW, 
            'Action': 'Set_Draw', 'Speed_Profile': 'Slow'
        })
        
        # B. วาด Segment (Draw)
        for x_pixel, y_pixel in segment[1:]:
            X_m = x_pixel * SCALING_FACTOR + X_OFFSET
            Y_m = y_pixel * SCALING_FACTOR + Y_OFFSET
            
            # คำสั่ง 3: Draw_Down (วาดตามเส้นทาง)
            final_ur5_trajectory.append({
                'X': X_m, 'Y': Y_m, 'Z': Z_DRAW, 
                'Roll': ROLL, 'Pitch': PITCH, 'Yaw': YAW, 
                'Action': 'Draw_Down', 'Speed_Profile': 'Draw'
            })
            
        # C. ยกปากกาเมื่อจบ Segment (Rapid Traverse)
        end_x_pixel, end_y_pixel = segment[-1]
        X_end_m = end_x_pixel * SCALING_FACTOR + X_OFFSET
        Y_end_m = end_y_pixel * SCALING_FACTOR + Y_OFFSET
        
        # คำสั่ง 4: Move_Lift (Pen Up) ณ จุดสิ้นสุด
        final_ur5_trajectory.append({
            'X': X_end_m, 'Y': Y_end_m, 'Z': Z_LIFT, 
            'Roll': ROLL, 'Pitch': PITCH, 'Yaw': YAW, 
            'Action': 'Move_Lift', 'Speed_Profile': 'Rapid'
        })

    return pd.DataFrame(final_ur5_trajectory)


#4. การทำงานหลัก (สมมติว่า edges_xy_flipped.csv มีข้อมูลอยู่)
# -----------------------------------------------------------------

# อ่านข้อมูลจากไฟล์ที่สร้างในขั้นตอน Edge Detection
try:
    coords = np.loadtxt("edges_xy_flipped.csv", delimiter=",", skiprows=1)
except FileNotFoundError:
    print("Error: ไม่พบไฟล์ 'edges_xy_flipped.csv' กรุณาทำขั้นตอน Edge Detection ก่อน")
    exit()

# 1. จัดระเบียบ Path เป็น Segments
path_segments = organize_and_segment_path(coords, MAX_GAP_PIXELS)

# 2. สร้าง Trajectory DataFrame
df_trajectory = generate_ur5_trajectory(path_segments)

# 3. บันทึกผลลัพธ์
df_trajectory.to_csv('ur5_trajectory_commands_optimized.csv', index=False)

print(f"✅ Path Planning เสร็จสมบูรณ์! ได้ {len(path_segments)} segments.")
print(f"ไฟล์ 'ur5_trajectory_commands_optimized.csv' พร้อมสำหรับ Inverse Kinematics")
print("---")
print("ตัวอย่าง 10 คำสั่งแรก:")
print(df_trajectory.head(10))

