#-----------------------------
#1. Image processing
#-----------------------------
"""This process is used to make the input image to a clear path, and prepare to trajectory generator
First, we use Canny Edge Detection to detect the edge of image the out put is csv. But there is some noise
and random pixels that unnecessary for detail, so we delete it.
    """
import cv2
import numpy as np

#import image and convert to grayscale and resize
img = cv2.imread("image/Temple.jpg",0)
# img = cv2.imread("image/FIBO.png",0)
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
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


result = edges
cv2.imshow("output_test", result)

canvas_width_m =
canvas_hight_m = 
img_height, img_width = img.shape[:2]
scale_x = canvas_width_m / img_width
scale_y = canvas_height_m / img_height


# kernel = np.ones((3,3), np.uint8)
# # Apply Canny Edge Detector
# blur = cv2.GaussianBlur(resized_image, (5, 5), 1.4)
# edges = cv2.Canny(blur, threshold1=100, threshold2=200)

# edges_dilated = cv2.dilate(edges, kernel, iterations=1)

# # 4. Finding contour
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # 5. วาดเส้น path (ทดสอบ)
# canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(canvas, contours, -1, (0,255,0), 1)

# #ดึงออกมาเป็น list ของจุดสำหรับ trajectory
# path_points = []
# for cnt in contours:
#     for point in cnt:
#         x, y = point[0]
#         path_points.append((x, y))

# # ให้หุ่นยนต์เคลื่อนผ่านแต่ละจุดในเวลาเท่ากัน
# t = np.linspace(0, len(path_points), len(path_points))
# # สร้าง trajectory (x(t), y(t))
# trajectory = [(t[i], path_points[i][0], path_points[i][1]) for i in range(len(t))]
# print(trajectory)
# # Display result
# #cv2.imshow('Oeiginal', imguse)
# cv2.imshow("Canny Edge Detection", edges_dilated)
# cv2.imshow("Path", canvas)

cv2.waitKey(0)
cv2.destroyAllWindows(0)