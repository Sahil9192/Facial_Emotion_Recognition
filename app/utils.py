import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (48, 48))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 48, 48, 1))
    return img_reshaped
