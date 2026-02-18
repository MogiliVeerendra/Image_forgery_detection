import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "casia")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("DATASET PATH:", DATASET_PATH)

# =====================================================
# PARAMETERS
# =====================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# =====================================================
# DATA AUGMENTATION
# =====================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
])

# =====================================================
# LOAD DATASET (AUTOMATIC CLASS DETECTION)
# =====================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# =====================================================
# NORMALIZATION
# =====================================================
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# =====================================================
# PERFORMANCE OPTIMIZATION
# =====================================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =====================================================
# CNN MODEL (BASELINE + AUGMENTATION)
# =====================================================
model = Sequential([
    data_augmentation,
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# =====================================================
# BUILD MODEL (IMPORTANT FIX)
# =====================================================
model.build(input_shape=(None, 224, 224, 3))

# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# SUMMARY
# =====================================================
model.summary()

# =====================================================
# TRAIN
# =====================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =====================================================
# SAVE MODEL
# =====================================================
MODEL_PATH = os.path.join(MODEL_DIR, "forgery_augmented_cnn.h5")
model.save(MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")
