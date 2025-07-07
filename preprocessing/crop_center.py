# preprocessing/crop_center.py
import cv2
import os

def crop_and_resize(img_path, size=(384, 384)):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    img_cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    img_resized = cv2.resize(img_cropped, size)
    return img_resized
