import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ===============================
# PROJECT PATHS
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "casia")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "forgery_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

print("DATASET PATH:", DATASET_PATH)
print("MODEL PATH:", MODEL_PATH)

# ===============================
# PARAMETERS
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5

# ===============================
# LOAD DATASET (SAFE METHOD)
# ===============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=["authentic", "tampered"],  # IMPORTANT
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=["authentic", "tampered"],
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ===============================
# NORMALIZE
# ===============================
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# ===============================
# CNN MODEL
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAIN
# ===============================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
# ===============================
# SAVE MODEL
# ===============================
model.save(MODEL_PATH)
print("✅ Model saved as:", MODEL_PATH)

