import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

from preprocess_and_load import load_data

# =========================
# LOAD DATA
# =========================
train_ds, val_ds = load_data()

# =========================
# DATA AUGMENTATION LAYER
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Apply augmentation ONLY to training data
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# =========================
# COMPUTE CLASS WEIGHTS
# =========================
class_counts = np.zeros(2)

for _, labels in train_ds:
    labels = labels.numpy().astype(int)
    class_counts[0] += np.sum(labels == 0)
    class_counts[1] += np.sum(labels == 1)

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
# TRAIN MODEL
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
model.save("../models/aug_weighted_forgery_model.h5")

with open("../models/aug_weighted_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Augmented + Weighted model training completed successfully")
