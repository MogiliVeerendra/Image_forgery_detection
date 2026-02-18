Continue generating the Image Forgery Detection project using the following environment:

Python Libraries:
- tensorflow==2.13.0
- keras==2.13.0
- opencv-python==4.8.0.76
- numpy==1.24.3
- matplotlib==3.7.2
- streamlit==1.28.0
- scikit-learn==1.3.1
- Pillow==10.0.0
-seaborn==0.12.2
-pandas==2.0.3  

Project Goal:
Build a CNN-based system to detect tampered vs authentic images using multiple datasets.

Important Constraints:
- Ensure compatibility with TensorFlow 2.13
- Use Keras Sequential API
- Avoid deprecated APIs
- Use .h5 model saving format
- Ensure paths work on Windows

Generate the following files inside /src:

1. preprocess.py
   - Image resizing to 224x224
   - Normalization
   - Error handling for corrupt images

2. preprocess_and_load.py
   - Load multiple datasets:
     ../datasets/casia
     ../datasets/columbia
     ../datasets/CoMoFoD_small_v2
   - Combine datasets
   - Split into training & validation

3. train_clean_model.py
   - Build CNN:
        Conv2D → ReLU → MaxPool (x3)
        Flatten → Dense → Sigmoid
   - Handle class imbalance
   - Save model to ../models/forgery_model.h5

4. evaluate_model.py
   - Load trained model
   - Compute:
        Accuracy
        Precision
        Recall
        Confusion Matrix
        ROC Curve & AUC

5. test_single_image.py
   - Allow user to browse image
   - Show prediction with confidence
   - Display image using OpenCV

6. web_app/app.py
   - Streamlit app
   - Upload image
   - Display prediction & confidence
   - Show image preview

Coding Rules:
- Use modular functions
- Add comments for each step
- Include exception handling
- Print useful logs

Expected Output:
✔ Trained model
✔ Evaluation metrics & ROC curve
✔ Single image prediction
✔ Streamlit web app
