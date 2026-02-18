train_clean_model.py
====================
Model training module for Image Forgery Detection project.

Responsibilities:
- Build CNN architecture with 3 convolutional blocks
- Handle class imbalance with class weights
- Train on combined datasets
- Save trained model to .h5 format
- Save training history and metrics
- Implement early stopping and model checkpointing

Author: Image Forgery Detection Team
Version: 1.0

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess_and_load import DatasetLoader
import matplotlib.pyplot as plt


class ForgeryCNN:
    """
    CNN model for image forgery detection.
    Architecture: 3 Conv blocks -> Flatten -> Dense layers -> Sigmoid output
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize ForgeryCNN model.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.training_history = None
        
    
    def build_model(self):
        """
        Build CNN architecture.
        
        Architecture:
        - Block 1: Conv(32) -> Conv(32) -> MaxPool -> Dropout(0.25)
        - Block 2: Conv(64) -> Conv(64) -> MaxPool -> Dropout(0.25)
        - Block 3: Conv(128) -> Conv(128) -> MaxPool -> Dropout(0.25)
        - Flatten -> Dense(512) -> Dropout(0.5) -> Dense(256) -> Dropout(0.5) -> Dense(1, Sigmoid)
        
        Returns:
            keras.Sequential: Compiled model
        """
        print("\n" + "="*60)
        print("BUILDING CNN MODEL")
        print("="*60)
        
        model = Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # ===== BLOCK 1 =====
            Conv2D(32, (3, 3), padding='same', name='conv1_1'),
            ReLU(name='relu1_1'),
            Conv2D(32, (3, 3), padding='same', name='conv1_2'),
            ReLU(name='relu1_2'),
            MaxPooling2D((2, 2), name='pool1'),
            Dropout(0.25, name='dropout1'),
            
            # ===== BLOCK 2 =====
            Conv2D(64, (3, 3), padding='same', name='conv2_1'),
            ReLU(name='relu2_1'),
            Conv2D(64, (3, 3), padding='same', name='conv2_2'),
            ReLU(name='relu2_2'),
            MaxPooling2D((2, 2), name='pool2'),
            Dropout(0.25, name='dropout2'),
            
            # ===== BLOCK 3 =====
            Conv2D(128, (3, 3), padding='same', name='conv3_1'),
            ReLU(name='relu3_1'),
            Conv2D(128, (3, 3), padding='same', name='conv3_2'),
            ReLU(name='relu3_2'),
            MaxPooling2D((2, 2), name='pool3'),
            Dropout(0.25, name='dropout3'),
            
            # ===== FLATTEN & DENSE LAYERS =====
            Flatten(name='flatten'),
            Dense(512, name='dense1'),
            ReLU(name='relu_dense1'),
            Dropout(0.5, name='dropout_dense1'),
            
            Dense(256, name='dense2'),
            ReLU(name='relu_dense2'),
            Dropout(0.5, name='dropout_dense2'),
            
            # ===== OUTPUT LAYER =====
            Dense(1, activation='sigmoid', name='output')
        ])
        
        self.model = model
        
        # Print model summary
        print("\n📐 Model Architecture:")
        print("-" * 60)
        model.summary()
        
        print("\n✅ Model built successfully!")
        return model
    
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate for Adam optimizer
        """
        print("\n" + "="*60)
        print("COMPILING MODEL")
        print("="*60)
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print(f"\n✅ Model compiled with:")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Binary Crossentropy")
        print(f"   Metrics: Accuracy, AUC")
    
    
def train(self, X_train, y_train, X_val, y_val, 
              class_weights=None, epochs=50, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
            class_weights (dict): Class weights for imbalance
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        
        Returns:
            keras.callbacks.History: Training history
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Create data augmentation generator
        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest'
        )
        
        print(f"\n🎨 Data Augmentation enabled")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {epochs}")
        
        # Create augmentation generator
        train_generator = datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            '../models/forgery_model_checkpoint.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=0
        )
        
        # Train model
        print(f"\n🚀 Starting training...")
        print("-" * 60)
        
        history = self.model.fit(
            train_generator,
            validation_data=(X_val, y_val),
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        self.training_history = history
        
        print("\n" + "-" * 60)
        print("✅ Training completed!")
        
        return history
    
    
def save_model(self, model_path='../models/forgery_model.h5'):
        """
        Save trained model to disk.
        
        Args:
            model_path (str): Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"\n💾 Saving model to {model_path}...")
        self.model.save(model_path)
        print(f"✅ Model saved successfully!")
        
        # Print model file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   File size: {file_size:.2f} MB")
    
    def save_training_history(self, history_path='../models/training_history.json'):
        """
        Save training history to JSON file.
        
        Args:
            history_path (str): Path to save history
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        print(f"\n💾 Saving training history to {history_path}...")
        
        # Convert history to serializable format
        history_dict = {}
        for key, value in self.training_history.history.items():
            history_dict[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"✅ Training history saved successfully!")
    
    def plot_training_history(self):
        """
        Plot training history (loss, accuracy, AUC).
        
        Returns:
            matplotlib.figure.Figure: Figure with plots
        """
        if self.training_history is None:
            print("⚠️  No training history available!")
            return None
        
        history = self.training_history.history
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot AUC
        axes[2].plot(history['auc'], label='Training AUC')
        axes[2].plot(history['val_auc'], label='Validation AUC')
        axes[2].set_title('AUC', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*70)
    print(" " * 15 + "IMAGE FORGERY DETECTION - TRAINING PIPELINE")
    print("="*70)
    
    try:
        # ===== STEP 1: Load datasets =====
        print("\n[STEP 1/5] Loading datasets...")
        loader = DatasetLoader(base_dataset_path="../datasets")
        images, labels, stats = loader.load_all_datasets()
        
        if len(images) == 0:
            print("❌ No images loaded! Exiting...")
            return
        
        # ===== STEP 2: Prepare data =====
        print("\n[STEP 2/5] Preparing data...")
        split_data = loader.split_dataset(test_size=0.2, val_size=0.1)
        class_weights = loader.get_class_weights()
        
        # ===== STEP 3: Build model =====
        print("\n[STEP 3/5] Building CNN model...")
        cnn = ForgeryCNN(input_shape=(224, 224, 3))
        cnn.build_model()
        cnn.compile_model(learning_rate=1e-4)
        
        # ===== STEP 4: Train model =====
        print("\n[STEP 4/5] Training model...")
        history = cnn.train(
            X_train=split_data['X_train'],
            y_train=split_data['y_train'],
            X_val=split_data['X_val'],
            y_val=split_data['y_val'],
            class_weights=class_weights,
            epochs=50,
            batch_size=32
        )
        
        # ===== STEP 5: Save model =====
        print("\n[STEP 5/5] Saving model and history...")
        cnn.save_model('../models/forgery_model.h5')
        cnn.save_training_history('../models/training_history.json')
        
        # Plot and save training history
        fig = cnn.plot_training_history()
        if fig:
            os.makedirs('../models', exist_ok=True)
            fig.savefig('../models/training_history_plot.png', dpi=100, bbox_inches='tight')
            print(f"✅ Training history plot saved to ../models/training_history_plot.png")
            plt.close(fig)
        
        print("\n" + "="*70)
        print(" " * 20 + "✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Model saved to: ../models/forgery_model.h5")
        print(f"📁 History saved to: ../models/training_history.json")
        print(f"📊 Plot saved to: ../models/training_history_plot.png")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()