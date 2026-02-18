import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np

from preprocess_and_load import load_data

# =========================
# LOAD DATA
# =========================
train_ds, val_ds = load_data()

# =========================
# COMPUTE CLASS WEIGHTS
# =========================
class_counts = np.zeros(2)

for _, labels in train_ds:
    labels = labels.numpy().astype(int)
    class_counts[0] += np.sum(labels == 0)  # Authentic
    class_counts[1] += np.sum(labels == 1)  # Tampered

total = class_counts.sum()
class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}

print("Class Weights:", class_weight)

# =========================
# MODEL DEFINITION
# =========================
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN WITH CLASS WEIGHTS
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight
)

# =========================
# SAVE MODEL & HISTORY
# =========================
model.save("../models/weighted_forgery_model.h5")

with open("../models/weighted_training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Weighted model training completed successfully")
