"""
preprocess_and_load.py
=======================
Dataset loading and preprocessing module for Image Forgery Detection project.

Responsibilities:
- Load images from multiple datasets (CASIA, Columbia, CoMoFoD)
- Combine datasets into single training dataset
- Split data into train, validation, and test sets
- Apply data augmentation (flip, rotation, zoom, brightness)
- Handle class imbalance

Author: Image Forgery Detection Team
Version: 1.0
"""

import os
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import load_images_from_directory, load_image


class DatasetLoader:
    """
    Handles loading and preprocessing of multiple forgery detection datasets.
    """
    
    def __init__(self, base_dataset_path="../datasets", target_size=(224, 224)):
        """
        Initialize DatasetLoader.
        
        Args:
            base_dataset_path (str): Base path to datasets directory
            target_size (tuple): Target image size (height, width)
        """
        self.base_dataset_path = base_dataset_path
        self.target_size = target_size
        self.all_images = []
        self.all_labels = []
        self.dataset_stats = {}
        
    
    def load_single_dataset(self, dataset_name, dataset_type="binary"):
        """
        Load a single dataset with authentic and tampered classes.
        
        Args:
            dataset_name (str): Name of dataset (casia, columbia, CoMoFoD_small_v2)
            dataset_type (str): Type of dataset structure
        
        Returns:
            tuple: (images, labels, stats)
        """
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_path = os.path.join(self.base_dataset_path, dataset_name)
        
        if not os.path.isdir(dataset_path):
            print(f"⚠️  Dataset path not found: {dataset_path}")
            return [], [], {}
        
        images = []
        labels = []
        stats = {
            'authentic': 0,
            'tampered': 0,
            'failed': 0,
            'total': 0
        }
        
        # Load authentic images (label: 0)
        authentic_path = os.path.join(dataset_path, 'authentic')
        if os.path.isdir(authentic_path):
            print(f"\n📁 Loading authentic images from {authentic_path}")
            auth_images, failed = load_images_from_directory(
                authentic_path, 
                self.target_size,
                verbose=True
            )
            
            for img, path in auth_images:
                images.append(img)
                labels.append(0)  # 0 = Authentic
            
            stats['authentic'] = len(auth_images)
            stats['failed'] += len(failed)
        else:
            print(f"⚠️  Authentic directory not found: {authentic_path}")
        
        # Load tampered images (label: 1)
        tampered_path = os.path.join(dataset_path, 'tampered')
        if os.path.isdir(tampered_path):
            print(f"\n📁 Loading tampered images from {tampered_path}")
            tam_images, failed = load_images_from_directory(
                tampered_path,
                self.target_size,
                verbose=True
            )
            
            for img, path in tam_images:
                images.append(img)
                labels.append(1)  # 1 = Tampered
            
            stats['tampered'] = len(tam_images)
            stats['failed'] += len(failed)
        else:
            print(f"⚠️  Tampered directory not found: {tampered_path}")
        
        stats['total'] = stats['authentic'] + stats['tampered']
        
        # Print dataset statistics
        print(f"\n📊 Dataset Statistics for {dataset_name}:")
        print(f"   Authentic images: {stats['authentic']}")
        print(f"   Tampered images: {stats['tampered']}")
        print(f"   Total loaded: {stats['total']}")
        print(f"   Failed to load: {stats['failed']}")
        
        return images, labels, stats
    
    
    def load_all_datasets(self):
        """
        Load all available datasets and combine them.
        
        Returns:
            tuple: (combined_images, combined_labels, dataset_stats)
        """
        print("\n" + "="*60)
        print("LOADING ALL DATASETS")
        print("="*60)
        
        all_images = []
        all_labels = []
        
        # List of datasets to load
        datasets_to_load = [
            'casia',
            'columbia',
            'CoMoFoD_small_v2'
        ]
        
        for dataset_name in datasets_to_load:
            images, labels, stats = self.load_single_dataset(dataset_name)
            
            all_images.extend(images)
            all_labels.extend(labels)
            self.dataset_stats[dataset_name] = stats
        
        # Convert to numpy arrays
        all_images = np.array(all_images, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)
        
        self.all_images = all_images
        self.all_labels = all_labels
        
        # Print combined statistics
        self._print_combined_stats()
        
        return all_images, all_labels, self.dataset_stats
    
    
    def _print_combined_stats(self):
        """Print combined statistics for all datasets."""
        print("\n" + "="*60)
        print("COMBINED DATASET STATISTICS")
        print("="*60)
        
        total_authentic = sum(stats['authentic'] for stats in self.dataset_stats.values())
        total_tampered = sum(stats['tampered'] for stats in self.dataset_stats.values())
        total_images = total_authentic + total_tampered;
        
        print(f"\n📊 Combined Statistics:")
        for dataset_name, stats in self.dataset_stats.items():
            print(f"\n   {dataset_name}:")
            print(f"      Authentic: {stats['authentic']}")
            print(f"      Tampered: {stats['tampered']}")
            print(f"      Total: {stats['total']}")
        
        print(f"\n   TOTAL:")
        print(f"      Authentic: {total_authentic}")
        print(f"      Tampered: {total_tampered}")
        print(f"      Combined Total: {total_images}")
        
        if total_images > 0:
            auth_pct = (total_authentic / total_images) * 100
            tam_pct = (total_tampered / total_images) * 100
            print(f"\n   Class Distribution:")
            print(f"      Authentic: {auth_pct:.2f}%")
            print(f"      Tampered: {tam_pct:.2f}%")
            
            # Check for imbalance
            if abs(auth_pct - 50) > 20:
                print(f"\n   ⚠️  Dataset is imbalanced!")
                print(f"      Class weights should be applied during training")
    
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            test_size (float): Fraction for test set
            val_size (float): Fraction for validation set (from training set)
            random_state (int): Random seed for reproducibility
        
        Returns:
            dict: Dictionary containing split datasets
        """
        print("\n" + "="*60)
        print("SPLITTING DATASET")
        print("="*60)
        
        if len(self.all_images) == 0:
            print("⚠️  No images loaded! Call load_all_datasets() first.")
            return None
        
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.all_images,
            self.all_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.all_labels
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_train_val
        )
        
        split_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Print split statistics
        self._print_split_stats(split_data)
        
        return split_data
    
    
    def _print_split_stats(self, split_data):
        """Print statistics about the data split."""
        print(f"\n📊 Data Split Statistics:")
        print(f"   Training set: {len(split_data['X_train'])} images")
        print(f"      Authentic: {np.sum(split_data['y_train'] == 0)}")
        print(f"      Tampered: {np.sum(split_data['y_train'] == 1)}")
        
        print(f"   Validation set: {len(split_data['X_val'])} images")
        print(f"      Authentic: {np.sum(split_data['y_val'] == 0)}")
        print(f"      Tampered: {np.sum(split_data['y_val'] == 1)}")
        
        print(f"   Test set: {len(split_data['X_test'])} images")
        print(f"      Authentic: {np.sum(split_data['y_test'] == 0)}")
        print(f"      Tampered: {np.sum(split_data['y_test'] == 1)}")
        
        total = len(split_data['X_train']) + len(split_data['X_val']) + len(split_data['X_test'])
        train_pct = (len(split_data['X_train']) / total) * 100
        val_pct = (len(split_data['X_val']) / total) * 100
        test_pct = (len(split_data['X_test']) / total) * 100
        
        print(f"\n   Percentages:")
        print(f"      Train: {train_pct:.2f}%")
        print(f"      Val: {val_pct:.2f}%")
        print(f"      Test: {test_pct:.2f}%")
    
    
    def get_class_weights(self):
        """
        Calculate class weights to handle imbalanced data.
        
        Returns:
            dict: Class weights {0: weight_authentic, 1: weight_tampered}
        """
        unique, counts = np.unique(self.all_labels, return_counts=True)
        total = np.sum(counts)
        
        class_weights = {}
        for cls, count in zip(unique, counts):
            weight = total / (len(unique) * count)
            class_weights[int(cls)] = weight
        
        print("\n📊 Class Weights (for imbalance handling):")
        print(f"   Authentic (0): {class_weights[0]:.4f}")
        print(f"   Tampered (1): {class_weights[1]:.4f}")
        
        return class_weights
    
    
    def create_augmentation_generator(self):
        """
        Create data augmentation generator.
        
        Returns:
            ImageDataGenerator: Configured augmentation generator
        """
        datagen = ImageDataGenerator(
            rotation_range=20,           # Random rotation
            horizontal_flip=True,        # Random horizontal flip
            vertical_flip=True,          # Random vertical flip
            zoom_range=0.2,              # Random zoom
            brightness_range=[0.8, 1.2], # Brightness adjustment
            width_shift_range=0.1,       # Width shift
            height_shift_range=0.1,      # Height shift
            fill_mode='nearest'
        )
        
        print("\n🎨 Data Augmentation Configuration:")
        print("   - Rotation: 20°")
        print("   - Horizontal Flip: Yes")
        print("   - Vertical Flip: Yes")
        print("   - Zoom: 0.2 (20%)")
        print("   - Brightness: 0.8 - 1.2")
        print("   - Width Shift: 0.1 (10%)")
        print("   - Height Shift: 0.1 (10%)")
        
        return datagen


def main():
    """
    Main function to test DatasetLoader
    """
    print("\n" + "="*60)
    print("Testing DatasetLoader")
    print("="*60)
    
    # Initialize loader
    loader = DatasetLoader(base_dataset_path="../datasets")
    
    # Load all datasets
    images, labels, stats = loader.load_all_datasets()
    
    # Get class weights
    class_weights = loader.get_class_weights()
    
    # Split dataset
    split_data = loader.split_dataset(test_size=0.2, val_size=0.1)
    
    # Create augmentation generator
    augmentation_gen = loader.create_augmentation_generator()
    
    print("\n✅ DatasetLoader test completed!")
    print("="*60)


if __name__ == "__main__":
    main()