import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

IMAGE_PATH = "image/FIBO.png"
OUTPUT_CSV = f"1_Trajectory.csv"

# Workspace
CANVAS_WIDTH_M = 0.8     # drawing width in meters
IMG_PROCESS_WIDTH = 600   # resize width pixels
MIN_CONTOUR_LEN = 15
VIA_POINT_DIST = 5*0.001   # downsample 5mm
SMOOTHING_FACTOR = 0.002 # spline smoothness

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

cv2.imshow("edge", edges)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


def downsample_points(points, min_dist):  #จำนวนจุดที่อยู่ห่างจากเพื่อนเล็กน้อย
    if len(points) < 2:
        return points
    kept = [points[0]]
    last = points[0]
    for i in range(1,len(points)-1):
        if np.linalg.norm(points[i]-last) >= min_dist:
            kept.append(points[i])
            last = points[i]
    kept.append(points[-1])
    return np.array(kept)




#close
cv2.waitKey(0)
cv2.destroyAllWindows()