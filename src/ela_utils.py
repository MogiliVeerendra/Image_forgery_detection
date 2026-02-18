from PIL import Image
import numpy as np
import cv2
import os

def convert_to_ela(image_path, quality=90):
    temp_file = "temp.jpg"
    original = Image.open(image_path).convert("RGB")
    original.save(temp_file, "JPEG", quality=quality)

    compressed = Image.open(temp_file)
    ela_image = np.abs(np.asarray(original) - np.asarray(compressed))

    ela_image = cv2.normalize(ela_image, None, 0, 255, cv2.NORM_MINMAX)
    os.remove(temp_file)

    return ela_image
