"""
preprocess.py
==============
Image preprocessing module for Image Forgery Detection project.

Responsibilities:
- Load images from file paths
- Resize images to 224x224
- Normalize pixel values to [0, 1]
- Handle corrupt or missing images
- Support multiple image formats (JPG, PNG, BMP, TIFF)

Author: Image Forgery Detection Team
Version: 1.0
"""

import cv2
import numpy as np
from pathlib import Path
import os


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size (height, width)
    
    Returns:
        np.ndarray: Preprocessed image array [0, 1] or None if failed
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be read or is corrupt
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image using OpenCV (BGR format)
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError: {e}")
        return None
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error loading image {image_path}: {e}")
        return None


def load_images_from_directory(directory_path, target_size=(224, 224), verbose=True):
    """
    Load all images from a directory.
    
    Args:
        directory_path (str): Path to directory containing images
        target_size (tuple): Target image size
        verbose (bool): Print loading progress
    
    Returns:
        list: List of tuples (image_array, image_path)
        list: List of failed image paths
    """
    images = []
    failed_images = []
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return images, failed_images
    
    # Get all image files
    image_files = [f for f in os.listdir(directory_path) 
                   if os.path.splitext(f.lower())[1] in supported_extensions]
    
    if verbose:
        print(f"📁 Found {len(image_files)} images in {directory_path}")
    
    # Load each image
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(directory_path, image_file)
        image = load_image(image_path, target_size)
        
        if image is not None:
            images.append((image, image_path))
        else:
            failed_images.append(image_path)
        
        if verbose and (idx + 1) % 10 == 0:
            print(f"   Loaded {idx + 1}/{len(image_files)} images...")
    
    if verbose:
        print(f"✅ Successfully loaded {len(images)} images")
        if failed_images:
            print(f"⚠️  Failed to load {len(failed_images)} images")
    
    return images, failed_images


def normalize_images(images_array):
    """
    Normalize image array to [0, 1] range.
    
    Args:
        images_array (np.ndarray): Array of images [batch, height, width, channels]
    
    Returns:
        np.ndarray: Normalized images
    """
    return images_array.astype(np.float32) / 255.0


def get_image_info(image_path):
    """
    Get information about an image.
    
    Args:
        image_path (str): Path to image
    
    Returns:
        dict: Image information or None
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width, channels = image.shape
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
        
        return {
            'path': image_path,
            'width': width,
            'height': height,
            'channels': channels,
            'size_mb': round(file_size, 2)
        }
    except Exception as e:
        print(f"❌ Error getting image info: {e}")
        return None


if __name__ == "__main__":
    """
    Test preprocess module
    """
    print("=" * 60)
    print("Testing preprocess.py module")
    print("=" * 60)
    
    # Test single image loading
    print("\n📝 Test 1: Single image loading")
    print("-" * 60)
    
    # Create a dummy test image for demonstration
    test_image_path = "test_image.png"
    
    # Create dummy image
    dummy_img = np.uint8(np.random.rand(300, 400, 3) * 255)
    cv2.imwrite(test_image_path, dummy_img)
    
    loaded_img = load_image(test_image_path)
    if loaded_img is not None:
        print(f"✅ Successfully loaded image")
        print(f"   Shape: {loaded_img.shape}")
        print(f"   Min value: {loaded_img.min():.4f}")
        print(f"   Max value: {loaded_img.max():.4f}")
    
    # Test image info
    print("\n📝 Test 2: Image information")
    print("-" * 60)
    img_info = get_image_info(test_image_path)
    if img_info:
        for key, value in img_info.items():
            print(f"   {key}: {value}")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("\n✅ All tests completed!")
    print("=" * 60)