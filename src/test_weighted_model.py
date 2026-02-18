import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess_and_load import load_data

# =========================
# LOAD WEIGHTED MODEL
# =========================
model = tf.keras.models.load_model("../models/weighted_forgery_model.h5")

# =========================
# LOAD DATA
# =========================
train_ds, val_ds = load_data()

y_true = []
y_pred = []

# =========================
# PREDICTION
# =========================
for images, labels in val_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# METRICS
# =========================
print("\n✅ WEIGHTED MODEL EVALUATION RESULTS\n")

print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Authentic", "Tampered"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
