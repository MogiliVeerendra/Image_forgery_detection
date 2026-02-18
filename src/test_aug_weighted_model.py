import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "../models/aug_weighted_forgery_model.h5"
DATASET_PATH = "../datasets/casia"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# LOAD TEST DATA
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)
print("\nAccuracy :", round(accuracy * 100, 2), "%\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
