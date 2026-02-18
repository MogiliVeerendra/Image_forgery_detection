# src/train_improved_model.py

import tensorflow as tf
from preprocess_and_load import load_data

EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_PATH = "../models/forgery_model.h5"

# -------------------------
# Load data
# -------------------------
train_ds, val_ds, class_weights = load_data()

# -------------------------
# Build model (EfficientNet)
# -------------------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False   # IMPORTANT for Limitation 1

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# -------------------------
# Compile
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train
# -------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# -------------------------
# Save model
# -------------------------
model.save(MODEL_PATH)
print("✅ Improved model trained & saved")
