import tensorflow as tf
import pickle
from preprocess_and_load import load_data
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "../models/aug_weighted_forgery_model.h5"
SAVE_MODEL_PATH = "../models/finetuned_forgery_model.h5"
SAVE_HISTORY_PATH = "../models/finetuned_history.pkl"

EPOCHS = 5
LEARNING_RATE = 1e-5
UNFREEZE_LAYERS = 40   # last 40 trainable layers

# -----------------------------
# LOAD DATA
# -----------------------------
train_ds, val_ds = load_data()

# -----------------------------
# COMPUTE CLASS WEIGHTS
# -----------------------------
labels = []
for _, y in train_ds:
    labels.extend(y.numpy().astype(int).flatten())

counter = Counter(labels)
total = sum(counter.values())

class_weights = {
    0: total / (2 * counter[0]),
    1: total / (2 * counter[1])
}

print("Class distribution:", counter)
print("Class weights:", class_weights)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# FREEZE ALL LAYERS FIRST
# -----------------------------
for layer in model.layers:
    layer.trainable = False

# -----------------------------
# UNFREEZE LAST N TRAINABLE LAYERS
# -----------------------------
trainable_layers = [
    layer for layer in model.layers if layer.weights
]

for layer in trainable_layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

print(f"Unfroze last {len(trainable_layers[-UNFREEZE_LAYERS:])} layers")

# -----------------------------
# RECOMPILE MODEL
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# FINE-TUNE MODEL
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# -----------------------------
# SAVE MODEL & HISTORY
# -----------------------------
model.save(SAVE_MODEL_PATH)

with open(SAVE_HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)

print("✅ Fine-tuning completed successfully")
