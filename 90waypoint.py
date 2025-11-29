import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- 1. CONFIGURATION ---
# IMAGE_PATH = "image/Bird.jpg" 
IMAGE_PATH = "image/FIBO.png"
OUTPUT_WAYPOINTS_CSV = "Waypoints_with_z.csv"

# --- Robot Workspace Settings ---
CENTER_X = 0.5              #m
CENTER_Y = 0.0              #m
DRAWING_WIDTH_M = 0.6       #m

Z_DRAW = 0.00               #m
Z_SAFE = 0.05               #m

IMG_PROCESS_HEIGHT = 400 # สมมติว่าความสูง 400

#Image size & 
IMG_PROCESS_WIDTH = 500     #px
MIN_PATH_PX = 10            #define new path
JUMP_THRESHOLD_M = 0.05     # กระโดดเกิน 1.0 pixel ในแกน X และ Y ถือว่าขึ้นเส้นให
SKIP_PIXEL_STEP = 1         #px reduce pixels

# --- 2. HELPER FUNCTIONS ---
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
    
    edges_bool = edges > 0
    skeleton_lee = skeletonize(edges_bool, method='lee')
    skeleton_uint8 = (skeleton_lee * 255).astype(np.uint8)
    
    return skeleton_uint8, new_h, target_width

# --- 3. MAIN PIPELINE ---
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
cv2.imwrite("img_edge_detection.png", edges)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Detected {len(contours)} contours.")

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
    
    # 3. วนลูปผ่าน Contours ทั้งหมด
    for path_id, cnt in enumerate(contours):
        # cnt เป็น NumPy array รูปร่าง (N, 1, 2)
        points = cnt.reshape(-1, 2) # แปลงเป็น (N, 2) == (px, py)
        
        # 4. วนลูปผ่านทุกจุดใน Contour
        for px, py in points:
            
            # 5. Coordinate Transformation (Mapping)
            # หาความแตกต่างจากจุดศูนย์กลาง
            shifted_x = px - pixel_center_x
            shifted_y = py - pixel_center_y
            
            # สูตรการแปลง: (ใช้ shifting_y (ภาพ) เพื่อกำหนดแกน X (หุ่นยนต์) 
            # และ shifting_x (ภาพ) เพื่อกำหนดแกน Y (หุ่นยนต์)
            rx = CENTER_X + (shifted_y * scale_factor) 
            ry = CENTER_Y + (shifted_x * scale_factor) 
            
            all_waypoints.append({
                'path_id': path_id,
                'meter_x': rx,
                'meter_y': ry
            })
            
    # คืนค่าเป็น DataFrame
    return pd.DataFrame(all_waypoints)

df_robot_coords = convert_contours_to_meter_coords(
    contours,
    IMG_PROCESS_WIDTH,
    IMG_PROCESS_HEIGHT,
    DRAWING_WIDTH_M,
    CENTER_X,
    CENTER_Y
)


df_robot_coords.to_csv("df_contour_m_FIBO.csv", index=False)