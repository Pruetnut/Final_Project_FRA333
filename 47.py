import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # ใช้สำหรับสุ่มสี
from mpl_toolkits.mplot3d import Axes3D #แสดงผล 3D
from scipy.interpolate import splprep, splev

IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = f"1_Trajectory.csv"

# Workspace
CANVAS_WIDTH_M = 0.8     # drawing width in meters
IMG_PROCESS_WIDTH = 500   #FIX
MIN_CONTOUR_LEN = 15
VIA_POINT_DIST = 5*0.001   # downsample 5mm
SMOOTHING_FACTOR = 0.002 # spline smoothness

#UR5 configuration
Z_SAFE = 0.01 #(m)
Z_DRAW = 0.0
commands = [] 


def process_image_to_edges(image_path, target_width):   #input is image out put is edge image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    aspect = target_width / w
    new_h = int(h * aspect)
    resized = cv2.resize(img, (target_width, new_h))
    blur = cv2.GaussianBlur(resized, (5,5),0)
    bilateral = cv2.bilateralFilter(blur,15,75,75)
    edges = cv2.Canny(bilateral,50,150)
    return edges, new_h, target_width

edges, img_h, img_w = process_image_to_edges(IMAGE_PATH, IMG_PROCESS_WIDTH)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
scale = CANVAS_WIDTH_M / img_w
offset_x = -CANVAS_WIDTH_M / 2  # ให้ 0,0 อยู่ตรงกลาง X
offset_y = (img_h * scale) / 2  # ให้ 0,0 อยู่ตรงกลาง Y

x_path, y_path, z_path = [], [], []
# เริ่มต้นที่จุด Home (0,0,Safe)
x_path.append(0); y_path.append(0); z_path.append(Z_SAFE)

for i, cnt in enumerate(contours):
    # cnt มีรูปร่างเป็น (N, 1, 2) ต้องดึงค่า X, Y ออกมา
    # cnt[:, 0, 0] คือ x ทั้งหมด
    # cnt[:, 0, 1] คือ y ทั้งหมด
    x = cnt[:, 0, 0]
    y = cnt[:, 0, 1]
    
    # พลอตเส้น
    plt.plot(x, y, linewidth=2, label=f'Line {i}')
    
    # (Optional) พลอตจุดเริ่มต้นของแต่ละเส้น จะได้รู้ว่าหุ่นยนต์เริ่มวาดจากไหน
    plt.plot(x[0], y[0], 'ro') # จุดสีแดงคือจุดเริ่ม
plt.gca().invert_yaxis() 
plt.axis('equal') 
# plt.show()

for contour in contours:
    # contour รูปร่างเป็น [[x1, y1], [x2, y2], ...] 
    points = contour.reshape(-1, 2)
    
    # 1. TRAVEL MOVE: ไปยังจุดแรกของเส้น (ต้องยกปากกาก่อนไป)
    start_x, start_y = points[0]
    commands.append({
        'action': 'MOVE',
        'x': start_x,
        'y': start_y,
        'z': Z_SAFE,  # ยกสูงไว้
        'note': 'Travel to start point'
    })
    
    # 2. PEN DOWN: กดปากกาลงที่จุดเดิม
    commands.append({
        'action': 'LOWER_PEN',
        'x': start_x,
        'y': start_y,
        'z': Z_DRAW,  # จรดกระดาษ
        'note': 'Pen Down'
    })
    
    # 3. DRAWING MOVE: ลากไปตามจุดที่เหลือ
    for i in range(1, len(points)):
        p_x, p_y = points[i]
        commands.append({
            'action': 'DRAW',
            'x': p_x,
            'y': p_y,
            'z': Z_DRAW, # ลากโดยที่ปากกายังจิ้มอยู่
            'note': 'Drawing line segment'
        })
        
    # 4. PEN UP: จบเส้นแล้ว ยกปากกาขึ้น
    last_x, last_y = points[-1]
    commands.append({
        'action': 'RAISE_PEN',
        'x': last_x,
        'y': last_y,
        'z': Z_SAFE, # ยกขึ้น
        'note': 'Pen Up (End of contour)'
    })

# ลองปริ้นดู 5 คำสั่งแรก
for cmd in commands:
    print(cmd)





for cnt in contours:
    points = cnt.reshape(-1, 2)
    start_x, start_y = points[0]
    
    # 2.1 TRAVEL: เคลื่อนจากจุดเดิม ไปยังจุดเริ่มของเส้นใหม่ (ลอยกลางอากาศ)
    # เพิ่มจุดปัจจุบันแต่ยกสูง
    x_path.append(x_path[-1])
    y_path.append(y_path[-1])
    z_path.append(Z_SAFE)
    
    # เคลื่อนไปจุดเริ่ม (ยังลอยอยู่)
    x_path.append(start_x)
    y_path.append(start_y)
    z_path.append(Z_SAFE)
    
    # 2.2 PEN DOWN: กดหัวปากกาลง
    x_path.append(start_x)
    y_path.append(start_y)
    z_path.append(Z_DRAW)
    
    # 2.3 DRAW: ลากเส้นตาม Contour (ติดพื้น)
    for p in points:
        x_path.append(p[0])
        y_path.append(p[1])
        z_path.append(Z_DRAW) # Z เป็น 0 ตลอดช่วงนี้
        
    # 2.4 PEN UP: จบเส้น ยกปากกาขึ้น
    end_x, end_y = points[-1]
    x_path.append(end_x)
    y_path.append(end_y)
    z_path.append(Z_SAFE)

# --- 3. แสดงผล 3D Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# แยกสีเพื่อความดูง่าย
# แปลงเป็น numpy array จะได้จัดการง่ายๆ
x_arr = np.array(x_path)
y_arr = np.array(y_path)
z_arr = np.array(z_path)

# เทคนิค: พลอตทีละช่วง
# ช่วงวาด (Z <= 0) ให้เป็นสีน้ำเงิน
# ช่วงยก (Z > 0) ให้เป็นสีแดง
for i in range(len(x_arr)-1):
    # ถ้าจุดนี้และจุดถัดไป Z เป็น 0 ทั้งคู่ แสดงว่ากำลังวาด
    if z_arr[i] == Z_DRAW and z_arr[i+1] == Z_DRAW:
        ax.plot(x_arr[i:i+2], y_arr[i:i+2], z_arr[i:i+2], color='blue', linewidth=2)
    else:
        # ถ้ามี Z สูง แสดงว่าเป็นช่วงยกหรือเคลื่อนย้าย
        ax.plot(x_arr[i:i+2], y_arr[i:i+2], z_arr[i:i+2], color='red', linestyle='--', linewidth=0.5, alpha=0.5)

ax.set_title("3D Toolpath Simulation (Red=Air Move, Blue=Draw)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z Height")

# กลับด้านแกน Y ให้ตรงกับภาพ
ax.invert_yaxis()
# มุมมองกล้อง (Elevation, Azimuth) ปรับเล่นได้
ax.view_init(elev=30, azim=-60) 

plt.show()




#close
cv2.waitKey(0)
cv2.destroyAllWindows()