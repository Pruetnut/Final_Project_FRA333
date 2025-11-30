import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- 1. CONFIGURATION ---
# IMAGE_PATH = "image/Bird.jpg" 
IMAGE_PATH = "image/FIBO.png"
OUTPUT_WAYPOINTS_CSV = "Matlab/Waypoints_from94e.csv"


#Image size &
IMG_PROCESS_WIDTH = 500     #px
SKIP_PIXEL_STEP = 1         #px reduce pixels
IMG_PROCESS_HEIGHT = 400 # สมมติว่าความสูง 400

# --- Robot Workspace Settings ---
CENTER_X = 0.5              #m
CENTER_Y = 0.0              #m
DRAWING_WIDTH_M = 0.6       #m

HOME_POS_X = 0.25            # ตำแหน่ง Home X
HOME_POS_Y = 0.25            # ตำแหน่ง Home Y
Z_DRAW = 0.00               #m
Z_SAFE = 0.05               #m


def process_image(path, target_width):
    img = cv2.imread(path, 0)
    if img is None:
        print(f"Warning: Image {path} not found. Using dummy square.")
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, 3)
    
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    
    #reduced noise
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    
    # edges_bool = edges > 0
    # skeleton_lee = skeletonize(edges_bool, method='lee')
    # skeleton_uint8 = (skeleton_lee * 255).astype(np.uint8)
    
    return edges, new_h, target_width

# --- SEE RESULT EDGE DETECTION ---
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
cv2.imwrite("img_edge_detection.png", edges)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Detected {len(contours)} contours.")

# contoursXY = []
# seen_coords = set()
# for path_id, cnt in enumerate(contours):
#     points = cnt.reshape(-1, 2)
#     for px, py in points:
#         current_coord = (px, py)
        # if current_coord not in seen_coords:
        #     seen_coords.add(current_coord)
#         contoursXY.append({
#             'x': px,
#             'y': py,
#             'z': 0,
#             'path': path_id
#         })
# df_contoursXY = pd.DataFrame(contoursXY)
# df_contoursXY.to_csv('contoursxy.csv', index=False)
# print(f"บันทึกข้อมูล {len(df_contoursXY)} จุด (ตัดจุดซ้ำออกแล้ว)")


def sort_contours_nearest(contours, start_pos=(0,0)):
    """
    จัดเรียงลำดับ Contours ตามระยะทางที่ใกล้ที่สุด (Greedy Search)
    Input: contours list จาก opencv
    Output: contours list ที่เรียงใหม่แล้ว
    """
    if not contours:
        return []

    # 1. เตรียมข้อมูล
    # เก็บข้อมูลว่าเส้นไหนใช้ไปแล้ว (visited) และจุดเริ่ม/จบของแต่ละเส้น
    cnt_data = []
    for cnt in contours:
        # กรองเส้นสั้นๆ ทิ้งไปเลยตั้งแต่ต้น (Noise Filter)
        if len(cnt) < 10: continue 
        
        pts = cnt.reshape(-1, 2)
        cnt_data.append({
            'pts': pts,
            'start': pts[0],      # จุดเริ่มของเส้น
            'end': pts[-1],       # จุดจบของเส้น
            'visited': False
        })
    
    if not cnt_data: return []

    sorted_cnts = []
    current_pos = np.array(start_pos) # เริ่มต้นที่ตำแหน่ง Home หรือ (0,0)

    # 2. วนลูปหาเส้นถัดไปเรื่อยๆ
    while True:
        nearest_idx = -1
        min_dist = float('inf')
        found_any = False

        # ไล่เช็คทุกเส้นที่ยังไม่ได้วาด
        for i, item in enumerate(cnt_data):
            if not item['visited']:
                found_any = True
                
                # คำนวณระยะห่างจาก ปลายปากกาล่าสุด -> หัวเส้นใหม่
                dist = np.linalg.norm(item['start'] - current_pos)
                
                # เก็บตัวที่ใกล้ที่สุดไว้
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
        
        # ถ้าไม่มีเส้นเหลือแล้ว ให้จบ
        if not found_any:
            break

        # 3. เพิ่มเส้นที่ได้รับเลือกลงใน List ใหม่
        cnt_data[nearest_idx]['visited'] = True
        sorted_cnts.append(cnt_data[nearest_idx]['pts']) # เก็บจุดดิบ
        
        # 4. อัปเดตตำแหน่งปลายปากกา เป็น "จุดจบ" ของเส้นที่เพิ่งเลือก
        current_pos = cnt_data[nearest_idx]['end']

    return sorted_cnts

sorted_contours = sort_contours_nearest(contours, start_pos=(0,0))

print(f"Sorted Contours: {len(sorted_contours)}")

def convert_contours_to_meter_coords(contours, IMG_PROCESS_WIDTH, IMG_PROCESS_HEIGHT, 
                                     DRAWING_WIDTH_M, CENTER_X, CENTER_Y):
    """
    แปลงพิกัด Pixel (จาก Contours) เป็นพิกัด Robot (เมตร)
    โดยใช้ Mapping Logic: Image Y (ลง) -> Robot X (ไปหน้า)
                           Image X (ขวา) -> Robot Y (ด้านข้าง)
    """
    
    # 1. คำนวณ Scale Factor
    scale_factor = DRAWING_WIDTH_M / IMG_PROCESS_WIDTH
    
    # 2. คำนวณจุดศูนย์กลางของภาพ (Pixel Center)
    pixel_center_x = IMG_PROCESS_WIDTH / 2
    pixel_center_y = IMG_PROCESS_HEIGHT / 2
    
    all_waypoints = []
    seen_coords = set()
    # 3. วนลูปผ่าน Contours ทั้งหมด
    for path_id, cnt in enumerate(contours):
        # cnt เป็น NumPy array รูปร่าง (N, 1, 2)
        points = cnt.reshape(-1, 2) # แปลงเป็น (N, 2) == (px, py)
        
        # 4. วนลูปผ่านทุกจุดใน Contour
        for px, py in points:
            current_coord = (px, py)
            if current_coord not in seen_coords:
                seen_coords.add(current_coord)
                # 5. Coordinate Transformation (Mapping)
                # หาความแตกต่างจากจุดศูนย์กลาง
                shifted_x = px - pixel_center_x
                shifted_y = py - pixel_center_y
                
                # สูตรการแปลง: (ใช้ shifting_y (ภาพ) เพื่อกำหนดแกน X (หุ่นยนต์) 
                # และ shifting_x (ภาพ) เพื่อกำหนดแกน Y (หุ่นยนต์)
                rx = CENTER_X - (shifted_y * scale_factor) 
                ry = CENTER_Y - (shifted_x * scale_factor) 
                
                all_waypoints.append({
                    'path_id': path_id,
                    'meter_x': rx,
                    'meter_y': ry
                })
        # path_id += 1
    # คืนค่าเป็น DataFrame
    return pd.DataFrame(all_waypoints)


#--- df_robot_coords = (path_id, meter_x, meter_y)
df_robot_coords = convert_contours_to_meter_coords(
    sorted_contours,
    IMG_PROCESS_WIDTH,
    IMG_PROCESS_HEIGHT,
    DRAWING_WIDTH_M,
    CENTER_X,
    CENTER_Y
)
#--- SAVE MATER CONTOURS XY ------------
print(f"Detected {len(df_robot_coords)} contours.")
# df_robot_coords.to_csv("df_contour_m_FIBO.csv", index=False)


def full_waypoints(xy_coords, Z_DRAW, Z_SAFE, HOME_POS_X, HOME_POS_Y):
    """
    สร้าง Waypoints XYZ สมบูรณ์ โดยเพิ่มจุดยก/วางปากกา (Type 0) เข้าไป 
    และกำหนดจุดเริ่มต้น/จุดสิ้นสุด (Home)
    
    Input: DataFrame ที่มี (path_id, meter_x, meter_y)
    Output: List of Dictionaries [{'x':..., 'y':..., 'z':..., 'type':...}]
    """
    
    # 1. กำหนด Global Home Point (จุดเดียวกับที่กำหนดไว้)
    HOME_POINT = np.array([HOME_POS_X, HOME_POS_Y, Z_SAFE])
    
    final_waypoints = []
    
    # --- GLOBAL START (จุดที่ 1) ---
    # 1. จุดเริ่มต้น (0.25,0.25,0.25) จุด home สร้างที่เริ่มจุดเดี่ยวจุดอรกใน dataframe
    # เราใช้ HOME_POS ที่กำหนดไว้
    final_waypoints.append({
        'x': HOME_POINT[0], 'y': HOME_POINT[1], 'z': HOME_POINT[2], 'type': 0, 'cmd': 'HOME_START', 
        'path_id': -1, 'count':0
    })
    
    # --- LOOP ผ่านแต่ละ Path ID ---
    
    for path_id, group in xy_coords.groupby('path_id'):
        
        # 2. เตรียมข้อมูล Path
        coords = group[['meter_x', 'meter_y']].values
        
        # P1: จุดแรกของ Path
        P1_xy = coords[0]
        
        # PN: จุดสุดท้ายของ Path
        PN_xy = coords[-1]
        
        # --- INJECT TRANSITION POINTS (Type 0) ---

        # 2. เพิ่มจุดแรก(x y คือจุดเเรกใน path นั้น) ปากกายกอยู่แล้ว z = Z_SAFE 
        # (Travel Approach - บินมาเหนือจุดเริ่ม)
        final_waypoints.append({
            'x': P1_xy[0], 'y': P1_xy[1], 'z': Z_SAFE, 'type': 0, 'cmd': 'TRAVEL_APPROACH', 
            'path_id': path_id
        })

        # 3. วงปากกาลงที่จุดแรก z = Z_DRAW (Pen Down)
        final_waypoints.append({
            'x': P1_xy[0], 'y': P1_xy[1], 'z': Z_DRAW, 'type': 0, 'cmd': 'PEN_DOWN', 
            'path_id': path_id
        })
        
        # --- DRAWING POINTS (Type 1) ---
        # 4. (รวมถึงจุดสุดท้ายของ Path ที่ระดับ Z_DRAW)
        # นำจุดทั้งหมดของ Path นั้นมาสร้าง Waypoints
        for x, y in coords:
            final_waypoints.append({
                'x': x, 'y': y, 'z': Z_DRAW, 'type': 1, 'cmd': 'DRAW_SEGMENT', 
                'path_id': path_id
            })
            
        # --- INJECT LIFT POINT (Type 0) ---
        # # 5. เตรียมตัวหยุดก่อนยกปากกา
        final_waypoints.append({
            'x': PN_xy[0], 'y': PN_xy[1], 'z': Z_DRAW, 'type': 0, 'cmd': 'PEN_DOWN', 
            'path_id': path_id
        })

        # 5. ยกปากกาที่จุดสุดท้าย ( xy อยู่ที่จุดสุดท้าย) z = Z_SAFE
        final_waypoints.append({
            'x': PN_xy[0], 'y': PN_xy[1], 'z': Z_SAFE, 'type': 0, 'cmd': 'LIFT_PEN', 'path_id': path_id
        })
        
    # --- GLOBAL END (จุดที่ 6) ---
    # 6. จุดสุดท้ายของ dataframe เป็นจุดเดี่ยวกับจุด Home
    final_waypoints.append({
        'x': HOME_POINT[0], 'y': HOME_POINT[1], 'z': HOME_POINT[2], 'type': 0, 'cmd': 'HOME_END', 
        'path_id': -1
    })
    
    # คืนค่าเป็น DataFrame
    return pd.DataFrame(final_waypoints)

waypoint_xyz = full_waypoints(df_robot_coords, Z_DRAW, Z_SAFE, HOME_POS_X, HOME_POS_Y)
waypoint_xyz.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print(f"Detected {len(waypoint_xyz)} waypoint.")

# --- LOAD DATA ---
# (สมมติว่าโค้ดได้อ่านไฟล์ INPUT_CSV มาแล้ว หรือสร้าง Dummy Path หากไม่พบ)

# [Code for loading/generating data as executed previously]
df = waypoint_xyz
# Prepare colors based on Type column
# Type 0 (Travel/Lift) = Red, Type 1 (Draw) = Blue
colors = np.where(df['type'] == 1, 'blue', 'red') 
sizes = np.where(df['type'] == 1, 5, 20) 

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter Plot for Points
ax.scatter(df['x'], df['y'], df['z'], c=colors, s=sizes, alpha=0.7)

# Plot connecting line (Sequence flow)
ax.plot(df['x'], df['y'], df['z'], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)


# --- Aesthetics and Labels ---
ax.set_title("3D Waypoint Path Visualization (Red=Travel, Blue=Draw)")
ax.set_xlabel("X (Forward) [m]")
ax.set_ylabel("Y (Side) [m]")
ax.set_zlabel("Z (Height) [m]")

# Ensure the Z-axis scale is compressed to match reality
ax.set_zlim(Z_DRAW - 0.01, Z_SAFE + 0.01) 
ax.set_box_aspect([1, 1, 0.2]) # Compress Z-axis for a table look

# Add point markers for Home/Start/End
ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], color='green', s=100, marker='*', label='Start Home')
ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], color='purple', s=100, marker='x', label='End Home')

ax.legend()
plt.show()