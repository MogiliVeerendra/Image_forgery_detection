import matplotlib.pyplot as plt
import pickle

# Load training history
with open("../models/history.pkl", "rb") as f:
    history = pickle.load(f)

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)

# -----------------------------
# Accuracy Plot
# -----------------------------
plt.figure()
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy", "Validation Accuracy"])
plt.grid(True)
plt.show()

# -----------------------------
# Loss Plot
# -----------------------------
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.grid(True)
plt.show()
