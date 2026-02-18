import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
from preprocess_and_load import load_data

# Load trained model (UPDATED NAME)
model = tf.keras.models.load_model("../models/forgery_model.h5")

# Load validation data
train_ds, val_ds = load_data()

y_true = []
y_pred = []

# Predict on validation dataset
for images, labels in val_ds:
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(predictions.flatten())

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Print evaluation results
print("\n✅ MODEL EVALUATION RESULTS\n")
print(f"Accuracy : {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Authentic", "Tampered"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
