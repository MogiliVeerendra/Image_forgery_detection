# Project: Image Forgery Detection using CNN

Build a deep learning-based image forgery detection system using TensorFlow/Keras.

The system should classify images as:
- Authentic
- Tampered

Follow the folder structure and responsibilities below.

--------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------

IMAGE_FORGERY_PROJECT/
│
├── datasets/
│   ├── casia/
│   ├── columbia/
│   └── CoMoFoD_small_v2/
│       ├── authentic/
│       └── tampered/
│
├── models/
│   └── (store trained .h5 models and history)
│
├── src/
│   ├── preprocess.py
│   ├── preprocess_and_load.py
│   ├── train_clean_model.py
│   ├── evaluate_model.py
│   ├── plot_training_graphs.py
│   ├── test_single_image.py
│
├── web_app/
│   └── app.py
│
└── requirements.txt

--------------------------------------------------
FUNCTIONAL REQUIREMENTS
--------------------------------------------------

1. Dataset Handling
- Load images from multiple datasets.
- Support authentic and tampered classes.
- Resize images to 224x224.
- Normalize pixel values.

2. Data Augmentation
- Random flip
- Rotation
- Zoom
- Brightness adjustment

3. CNN Model
- 3 Convolution blocks (Conv2D + ReLU + MaxPooling)
- Flatten layer
- Dense layer
- Sigmoid output for binary classification

4. Training
- Train on combined datasets.
- Use class weights for imbalance.
- Save model to /models/forgery_model.h5
- Save training history.

5. Evaluation
- Calculate accuracy, precision, recall.
- Plot ROC curve and compute AUC.
- Display confusion matrix.

6. Single Image Testing
- Allow user to select image from file dialog.
- Show prediction with confidence.
- Display image with label.

7. Web App (Streamlit)
- Upload image.
- Show prediction (Authentic or Tampered).
- Display confidence score.
- Show image preview.

--------------------------------------------------
TECHNOLOGIES
--------------------------------------------------

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- scikit-learn

--------------------------------------------------
CODING GUIDELINES
--------------------------------------------------

- Modular code.
- Each file should handle one responsibility.
- Use reusable functions.
- Include error handling for missing images.
- Ensure paths work relative to project root.

--------------------------------------------------
EXPECTED OUTPUT
--------------------------------------------------

✔ Trained CNN model
✔ ROC and training graphs
✔ Single image prediction tool
✔ Streamlit web app for forgery detection
