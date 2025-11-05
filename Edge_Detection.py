import cv2
import numpy as np

kernel = np.ones((3,3), np.uint8)

#import image and convert to grayscale and resize
img = cv2.imread("image/Bird.jpg",0)
imgeq = cv2.resize(img,(1000,700))

# Apply Canny Edge Detector
blur = cv2.GaussianBlur(imgeq, (5, 5), 1.4)
edges = cv2.Canny(blur, threshold1=100, threshold2=200)

edges_dilated = cv2.dilate(edges, kernel, iterations=1)

# 4. Finding contour
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 5. วาดเส้น path (ทดสอบ)
canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas, contours, -1, (0,255,0), 1)

#ดึงออกมาเป็น list ของจุดสำหรับ trajectory
path_points = []
for cnt in contours:
    for point in cnt:
        x, y = point[0]
        path_points.append((x, y))

# ให้หุ่นยนต์เคลื่อนผ่านแต่ละจุดในเวลาเท่ากัน
t = np.linspace(0, len(path_points), len(path_points))
# สร้าง trajectory (x(t), y(t))
trajectory = [(t[i], path_points[i][0], path_points[i][1]) for i in range(len(t))]
print(trajectory)
# Display result
#cv2.imshow('Oeiginal', imguse)
cv2.imshow("Canny Edge Detection", edges_dilated)
cv2.imshow("Path", canvas)

cv2.waitKey(0)
cv2.destroyAllWindows(0)