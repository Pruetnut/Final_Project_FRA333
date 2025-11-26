# -------------------- 0. PARAMETERS --------------------
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
#-----------------------------
#1. Image processing
#-----------------------------
"""This process is used to make the input image to a clear path, and prepare to trajectory generator
First, we use Canny Edge Detection to detect the edge of image the out put is csv. But there is some noise
and random pixels that unnecessary for detail, so we delete it.
    """
import cv2
import numpy as np
import matplotlib.pyplot as plt

#image path 
input_image = "image/FIBO.png"
# input_image = "image/Temple.jpg"
img = cv2.imread(input_image)

def process_image_to_edges(image_path):                 #output is an image of edge
    #import image and convert to grayscale and resize
    img = cv2.imread(image_path,0)
    original_height, original_width = img.shape[:2]
    new_width = 200
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    
    #--- use double blur to simplify the line form edge detection---
    g_blurred = cv2.GaussianBlur(resized_image, (5,5),0)
    bilateral = cv2.bilateralFilter(g_blurred, 15, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    return edges
    
edge_image = process_image_to_edges(input_image)
cv2.imshow("output_test", edge_image)

#--- make image pixels fit to work space in meter
if edge_image is not None:
    canvas_width_m = 500/1000       #work space drawing width(m)
    canvas_height_m = 500/1000      #work space drawing height(m)
    img_height, img_width = edge_image.shape[:2]
    scale_x = canvas_width_m / img_width
    scale_y = canvas_height_m / img_height
    #---------------------------
    # Step 2 Contour Extraction
    #----------------------------
    def extract_contours_as_paths(edge_image, min_area=100):
        """
            Step 2: Contour Extraction
            หน้าที่: แปลงภาพขาวดำ (Pixels) ให้เป็นเส้น Path (List of points)
        """
    # cv2.RETR_LIST: ดึงเส้นมาทั้งหมดโดยไม่สนความซับซ้อนของแม่-ลูก
    # cv2.CHAIN_APPROX_SIMPLE: เก็บเฉพาะจุดเลี้ยว (ประหยัดเมม) หรือใช้ CHAIN_APPROX_NONE ถ้าอยากได้ละเอียดทุกจุด
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        valid_paths = []

        for cnt in contours:
            # กรอง Noise: ถ้าเส้นสั้นเกินไป หรือเป็นจุดเล็กๆ ไม่ต้องเอามา
            if cv2.contourArea(cnt) > min_area or cv2.arcLength(cnt, False) > 50:
                
                # (Optional) Smooth Path: ลดจำนวนจุดลงเพื่อให้เส้นเรียบขึ้น หุ่นยนต์เดินนิ่งขึ้น
                epsilon = 0.001 * cv2.arcLength(cnt, False) # ค่าความละเอียด (ยิ่งมาก เส้นยิ่งเหลี่ยม)
                approx = cv2.approxPolyDP(cnt, epsilon, False)
                
                # reshape จาก (N, 1, 2) เป็น (N, 2) ให้ใช้ง่ายๆ
                valid_paths.append(approx.reshape(-1, 2))
                
        return valid_paths

#show result of find contour
def visualize_paths_matplotlib(paths):
    """
    พล็อตกราฟเส้นทางเดิน (Flip Y ให้เหมือนรูปภาพ)
    """
    plt.figure(figsize=(10, 8))
    
    for i, path in enumerate(paths):
        # path[:, 0] คือ x ทั้งหมด, path[:, 1] คือ y ทั้งหมด
        x = path[:, 0]
        y = path[:, 1]
        
        # พล็อตเส้น
        plt.plot(x, y, marker='.', markersize=1, label=f'Line {i}')
        
    # *** สำคัญ: ต้องกลับด้านแกน Y ***
    # เพราะรูปภาพ (0,0) อยู่มุมบนซ้าย แต่กราฟ (0,0) อยู่มุมล่างซ้าย
    plt.gca().invert_yaxis()
    
    plt.title(f"Robot Path Visualization ({len(paths)} lines)")
    plt.axis('equal') # สัดส่วนแกน X, Y เท่ากัน (รูปไม่เบี้ยว)
    plt.show()

paths = extract_contours_as_paths(edge_image)
visualize_paths_matplotlib(paths)

#-----------------------------
# 3. Coordinate Transformation & Optimization
#-----------------------------
import math

# กำหนดจุดเริ่มวาด (Offset) เทียบกับฐานหุ่นยนต์ (Base Frame)
# สมมติ: กระดาษวางห่างจากฐานไปทางแกน X 0.4 เมตร, และกึ่งกลางกระดาษอยู่ที่ Y=0
OFFSET_X = 0.4 
OFFSET_Y = -0.25  # (เริ่มที่มุมซ้ายของกระดาษ สมมติกระดาษกว้าง 0.5m เริ่มที่ -0.25)

def transform_and_sort_paths(pixel_paths, scale_x, scale_y, img_height, offset_x, offset_y):
    """
    Step 3 & 4: แปลงพิกัด Pixel -> Meter และเรียงลำดับเส้น
    """
    robot_paths = []
    
    # --- 3.1 แปลง Pixel เป็น Meter ---
    for path in pixel_paths:
        new_path = []
        for point in path:
            u, v = point # u=pixel_x, v=pixel_y
            
            # แปลงหน่วย
            x_meter = u * scale_x
            y_meter = v * scale_y
            
            # Apply Offset & Flip Y
            # หุ่นยนต์: Y มักจะชี้ไปทางซ้าย/ขวา, แต่รูปภาพ Y ชี้ลง
            # สูตร: Y_robot = (Height_meter - y_meter) + Offset_Y (เพื่อกลับหัว)
            # หรือถ้า Map ตรงๆ:
            final_x = x_meter + offset_x
            
            # ต้องดูว่าหุ่นยนต์คุณตั้งแกนยังไง อันนี้สมมติมาตรฐาน:
            # กลับด้านแกน Y ของรูปภาพ (เพราะรูปภาพ (0,0) อยู่บนซ้าย)
            final_y = ((img_height * scale_y) - y_meter) + offset_y 
            
            new_path.append([final_x, final_y])
        robot_paths.append(new_path)

    # --- 3.2 เรียงลำดับเส้น (Greedy / Nearest Neighbor) ---
    if not robot_paths:
        return []

    sorted_paths = []
    unvisited = robot_paths.copy()
    current_pos = (offset_x, offset_y) # เริ่มต้นที่จุด Home ของกระดาษ

    while unvisited:
        nearest_index = -1
        min_dist = float('inf')

        # หาเส้นที่จุดเริ่มต้นอยู่ใกล้หัวปากกาที่สุด
        for i, path in enumerate(unvisited):
            start_point = path[0]
            dist = math.sqrt((current_pos[0] - start_point[0])**2 + (current_pos[1] - start_point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        
        # เลือกเส้นนั้น
        nearest_path = unvisited.pop(nearest_index)
        sorted_paths.append(nearest_path)
        
        # อัปเดตตำแหน่งล่าสุดเป็นจุดจบของเส้นที่เพิ่งวาด
        current_pos = nearest_path[-1]

    return sorted_paths

# เรียกใช้งานฟังก์ชัน
final_robot_paths = transform_and_sort_paths(
    paths, 
    scale_x, 
    scale_y, 
    img_height, 
    OFFSET_X, 
    OFFSET_Y
)

#-----------------------------
# แสดงผลลัพธ์สุดท้าย (ในหน่วยเมตร)
#-----------------------------
print(f"Total paths: {len(final_robot_paths)}")
if len(final_robot_paths) > 0:
    print(f"Sample Point (Meter): {final_robot_paths[0][0]}")

# Visualize ดูลำดับการวาด (สีจะไล่ตามลำดับ)
plt.figure(figsize=(8, 8))
for i, path in enumerate(final_robot_paths):
    path = np.array(path)
    # ใช้ colormap เพื่อดูว่าเส้นไหนวาดก่อน-หลัง (ม่วง -> เหลือง)
    color = plt.cm.viridis(i / len(final_robot_paths)) 
    plt.plot(path[:, 0], path[:, 1], marker='.', markersize=2, color=color)

plt.title("Final Robot Trajectories (Meters) & Draw Order")
plt.xlabel("X Robot (m)")
plt.ylabel("Y Robot (m)")
plt.axis('equal')
plt.grid(True)
plt.show()# 3. Coordinate Transformation & Optimization
#-----------------------------
import math

# กำหนดจุดเริ่มวาด (Offset) เทียบกับฐานหุ่นยนต์ (Base Frame)
# สมมติ: กระดาษวางห่างจากฐานไปทางแกน X 0.4 เมตร, และกึ่งกลางกระดาษอยู่ที่ Y=0
OFFSET_X = 0.4 
OFFSET_Y = -0.25  # (เริ่มที่มุมซ้ายของกระดาษ สมมติกระดาษกว้าง 0.5m เริ่มที่ -0.25)



cv2.waitKey(0)
cv2.destroyAllWindows(0)