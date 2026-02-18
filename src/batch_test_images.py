import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from tkinter import filedialog, Tk

MODEL_PATH = "../models/forgery_model.h5"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

def select_folder():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select Image Folder")

def predict_folder(folder_path):
    results = []

    print("\n🔍 Batch Testing Started...\n")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)[0][0]
            confidence = pred if pred > 0.5 else 1 - pred
            label = "FORGED" if pred > 0.5 else "AUTHENTIC"

            print(f"{file:30} → {label} ({confidence*100:.2f}%)")
            results.append([file, label, f"{confidence*100:.2f}%"])

    return results

def save_csv(results):
    with open("batch_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Prediction", "Confidence"])
        writer.writerows(results)

    print("\n✅ Results saved as batch_results.csv")

if __name__ == "__main__":
    folder = select_folder()
    if folder:
        results = predict_folder(folder)
        save_csv(results)
