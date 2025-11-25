#-----------------------------
#1. Image processing
#-----------------------------
"""This process is used to make the input image to a clear path, and prepare to trajectory generator
First, we use Canny Edge Detection to detect the edge of image the out put is csv. But there is some noise
and random pixels that unnecessary for detail, so we delete it.
    """
import cv2
import numpy as np

#image path 
input_image = "image/FIBO.png"
img = cv2.imread(input_image)

def process_image_to_edges(image_path):
    #import image and convert to grayscale and resize
    img = cv2.imread(image_path,0)
    original_height, original_width = img.shape[:2]
    new_width = 600
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    #--- use double blur to simplify the line form edge detection---
    g_blurred = cv2.GaussianBlur(resized_image, (5,5),0)
    bilateral = cv2.bilateralFilter(g_blurred, 15, 75, 75)
    #--- Edge Detection --------------------------------------------
    edges = cv2.Canny(bilateral, 50, 150)
    return edges
    
edge_image = process_image_to_edges(input_image)
cv2.imshow("output_test", edge_image)

#--- make image pixels fit to work space in meter
canvas_width_m = 500/1000       #work space width(m)
canvas_height_m = 500/1000      #work space height(m)
img_height, img_width = edge_image.shape[:2]
scale_x = canvas_width_m / img_width
scale_y = canvas_height_m / img_height

#-----------------------------
# 2. contour
#-----------------------------
def extract_contours_as_paths(edge_image, min_area=10):
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

def transform_pixels_to_robot_frame(paths, img_dims, robot_dims, offset):
    """
    Step 3: Coordinate Transformation (สำคัญมาก!)
    หน้าที่: แปลงพิกัด Pixel (u, v) เป็นพิกัดหุ่นยนต์ (x, y) หน่วยเมตร
    """
    img_w, img_h = img_dims
    robot_w, robot_h = robot_dims     # พื้นที่กระดาษจริง (เช่น 0.2 เมตร x 0.2 เมตร)
    offset_x, offset_y = offset       # จุดเริ่มวาดเทียบกับฐานหุ่น
    
    robot_paths = []

    for path in paths:
        new_path = []
        for point in path:
            u, v = point # u = pixel x, v = pixel y (ระวัง! v นับจากบนลงล่าง)
            
            # 1. Scale: แปลง Pixel เป็น Meter
            # สูตร: (ค่า pixel / ความกว้างรูปทั้งหมด) * ความกว้างจริง
            x_meter = (u / img_w) * robot_w
            y_meter = (v / img_h) * robot_h
            
            # 2. Flip Y-Axis & Offset
            # ภาพ: (0,0) อยู่มุมซ้ายบน, Y ลงล่างเป็นบวก
            # หุ่นยนต์: ปกติ Y ขึ้นบนเป็นบวก หรือตาม World Frame ที่ตั้งไว้
            # กรณีนี้สมมติให้กลับด้าน Y เพื่อให้ภาพไม่กลับหัว
            final_x = x_meter + offset_x
            final_y = (robot_h - y_meter) + offset_y # Flip Y
            
            new_path.append((final_x, final_y))
        
        robot_paths.append(new_path)
        
    return robot_paths


# --- ส่วนการเรียกใช้งาน (Main Simulation) ---
if __name__ == "__main__":
    # Parameters
    INPUT_IMAGE = input_image
    PAPER_SIZE = (0.20, 0.15) # กว้าง 20cm, สูง 15cm (หน่วยเมตร)
    ROBOT_OFFSET = (0.3, -0.1) # หุ่นเริ่มวาดที่ X=0.3m, Y=-0.1m จากฐาน
    
    # 1. Process Image
    edges, dims = process_image_to_edges(INPUT_IMAGE)
    
    if edges is not None:
        # 2. Get Paths (Pixels)
        pixel_paths = extract_contours_as_paths(edges)
        print(f"Found {len(pixel_paths)} contours in image.")
        
        # 3. Convert to Robot Coordinates
        robot_trajectories_xy = transform_pixels_to_robot_frame(pixel_paths, dims, PAPER_SIZE, ROBOT_OFFSET)
        
        # ตัวอย่างผลลัพธ์เส้นแรก จุดแรก
        print(f"Start Point of Line 1 (Robot Frame): {robot_trajectories_xy[0][0]}")
        print("ข้อมูลนี้พร้อมส่งไป Trajectory Generator แล้ว!")

contour_path = extract_contours_as_paths(edge_image)
cv2.drawContours(img, contour_path, -1, (0,255,0), 3)


cv2.waitKey(0)
cv2.destroyAllWindows(0)