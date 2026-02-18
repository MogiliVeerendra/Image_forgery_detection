import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("../models/forgery_model.h5")

dummy = np.random.rand(1, 224, 224, 3).astype("float32")
pred = model.predict(dummy)

print("Raw output:", pred)
