import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
IMAGE_PATH = "image/FIBO.png" 
OUTPUT_WAYPOINTS_CSV = "Waypoints_For_Ruckig.csv"

# Robot Config
CENTER_X = 0.5; CENTER_Y = 0.0; DRAWING_WIDTH_M = 0.8
Z_DRAW = 0.00 # พื้น
IMG_PROCESS_WIDTH = 500
SKIP_PIXEL_STEP = 4 

# --- HELPER FUNCTIONS ---
def process_image(path, target_width):
    img = cv2.imread(path, 0)
    if img is None:
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, 3)
    
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    bilateral = cv2.bilateralFilter(blur, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    
    edges_bool = edges > 0
    skeleton = skeletonize(edges_bool, method='lee')
    return (skeleton * 255).astype(np.uint8), new_h, target_width

def sort_contours_nearest(contours, start_pos=(0,0)):
    if not contours: return []
    sorted_cnts = []
    cnt_data = []
    for cnt in contours:
        if len(cnt) < 2: continue
        pts = cnt.reshape(-1, 2)
        cnt_data.append({'pts': pts, 'start': pts[0], 'end': pts[-1], 'visited': False})
    
    current_pos = np.array(start_pos)
    while True:
        nearest_idx = -1; min_dist = float('inf'); found_any = False
        for i, item in enumerate(cnt_data):
            if not item['visited']:
                found_any = True
                dist = np.linalg.norm(item['start'] - current_pos)
                if dist < min_dist: min_dist = dist; nearest_idx = i
        if not found_any: break
        cnt_data[nearest_idx]['visited'] = True
        sorted_cnts.append(cnt_data[nearest_idx]['pts'])
        current_pos = cnt_data[nearest_idx]['end']
    return sorted_cnts

# --- MAIN PIPELINE ---
edges, img_h, img_w = process_image(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_contours = sort_contours_nearest(contours, start_pos=(0,0))

scale_factor = DRAWING_WIDTH_M / img_w
pixel_center_x = img_w / 2
pixel_center_y = img_h / 2

def to_robot(px, py):
    rx = CENTER_X + ((py - pixel_center_y) * scale_factor) 
    ry = CENTER_Y + ((px - pixel_center_x) * scale_factor) 
    return rx, ry

waypoints = []

# จุด Home (Type 0)
# waypoints.append({'x': 0.3, 'y': 0.0, 'z': 0.09, 'type': 0, 'cmd': 'HOME'})
HOME_X = 0.2
HOME_Y = 0.0
HOME_Z = 0.02 # เอาสูงกว่า Z_SAFE หน่อยก็ได้เพื่อความปลอดภัยตอนจบ

# เพิ่มจุดสุดท้ายลงใน List
waypoints.append({
    'x': HOME_X, 
    'y': HOME_Y, 
    'z': HOME_Z, 
    'type': 0,       # Type 0 = สั่งให้ File 2 สร้าง Arch บินมาหาจุดนี้
    'cmd': 'HOME_END'
})

for raw_points in sorted_contours:
    points = raw_points[::SKIP_PIXEL_STEP]
    if not np.array_equal(points[-1], raw_points[-1]):
        points = np.vstack((points, raw_points[-1]))
    if len(points) < 2: continue

    # --- LOGIC ใหม่: ไม่สร้าง Lift/Lower เองแล้ว ---
    
    # จุดแรกของเส้น -> Type 0 (บอกว่าเป็นจุดเริ่มเส้นใหม่ เดี๋ยว File 2 สร้าง Arch มาหาเอง)
    sx, sy = to_robot(points[0][0], points[0][1])
    waypoints.append({'x': sx, 'y': sy, 'z': Z_DRAW, 'type': 0, 'cmd': 'START_LINE'})
    
    # จุดที่เหลือ -> Type 1 (วาดต่อเนื่อง)
    for p in points[1:]:
        rx, ry = to_robot(p[0], p[1])
        waypoints.append({'x': rx, 'y': ry, 'z': Z_DRAW, 'type': 1, 'cmd': 'DRAW'})
        
# กำหนดพิกัด Home ที่ต้องการ (ควรยกสูง Z_SAFE หรือสูงกว่า)
HOME_X = 0.25
HOME_Y = 0.0
HOME_Z = 0.02 # เอาสูงกว่า Z_SAFE หน่อยก็ได้เพื่อความปลอดภัยตอนจบ

# เพิ่มจุดสุดท้ายลงใน List
waypoints.append({
    'x': HOME_X, 
    'y': HOME_Y, 
    'z': HOME_Z, 
    'type': 0,       # Type 0 = สั่งให้ File 2 สร้าง Arch บินมาหาจุดนี้
    'cmd': 'HOME_END'
})

# Save
df = pd.DataFrame(waypoints)
df.to_csv(OUTPUT_WAYPOINTS_CSV, index=False)
print(f"Saved {len(df)} points. Clean sequence (0=Start, 1=Draw).")

# Plot Preview
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], c=df['type'], cmap='bwr')
plt.show()