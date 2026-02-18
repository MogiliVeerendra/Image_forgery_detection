# src/config.py

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DATASETS = [
    "../datasets/casia",
    "../datasets/columbia",
    "../datasets/CoMoFoD_small_v2"
]

MODEL_PATH = "../models/forgery_model.h5"

CLASS_NAMES = ["authentic", "tampered"]
