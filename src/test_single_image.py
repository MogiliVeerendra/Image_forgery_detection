"""
Image Forgery Detection - Single Image Testing Module
=====================================================
Interactive GUI for testing single images with real-time predictions.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_image


class SingleImageTester:
    """
    Interactive single image testing class.
    Allows users to select and test images with GUI interface.
    """
    
    def __init__(self, model_path='../models/forgery_model.h5'):
        """
        Initialize SingleImageTester.
        
        Args:
            model_path (str): Path to trained .h5 model
        """
        print("\n" + "="*70)
        print("INITIALIZING SINGLE IMAGE TESTER")
        print("="*70)
        
        self.model_path = model_path
        self.model = None
        self.image_path = None
        self.image = None
        self.image_original = None
        self.prediction = None
        self.confidence = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """
        Load trained model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        print(f"\n[STEP 1/4] Loading model from: {self.model_path}")
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}\n"
                    f"Please train the model first using: python train_clean_model.py"
                )
            
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"   Model type: {type(self.model).__name__}")
            print(f"   Total parameters: {self.model.count_params():,}")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR loading model: {str(e)}")
            sys.exit(1)
    
    def select_image_gui(self):
        """
        Open file dialog to select image.
        
        Returns:
            str: Path to selected image or None
        """
        print(f"\n[STEP 2/4] Opening file dialog...")
        
        try:
            # Create hidden root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select an Image for Forgery Detection",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("JPG files", "*.jpg *.jpeg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                ]
            )
            
            root.destroy()
            
            if file_path:
                print(f"✅ Image selected: {file_path}")
                self.image_path = file_path
                return file_path
            else:
                print(f"⚠️ No image selected")
                return None
                
        except Exception as e:
            print(f"❌ ERROR opening file dialog: {str(e)}")
            return None
    
    def load_image_file(self, image_path=None):
        """
        Load and preprocess image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (processed_image, original_image) or (None, None)
        """
        print(f"\n[STEP 3/4] Loading and processing image...")
        
        if image_path is None:
            image_path = self.image_path
        
        if image_path is None:
            print(f"❌ ERROR: No image path provided")
            return None, None
        
        try:
            # Load and preprocess image
            processed_image = load_image(image_path)
            
            # Load original for display
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            if processed_image is None or original_image is None:
                raise ValueError("Failed to load image")
            
            self.image = processed_image
            self.image_original = original_image
            
            print(f"✅ Image loaded successfully!")
            print(f"   Processed shape: {processed_image.shape}")
            print(f"   Original shape: {original_image.shape}")
            print(f"   Pixel range: [{processed_image.min():.4f}, {processed_image.max():.4f}]")
            
            return processed_image, original_image
            
        except FileNotFoundError:
            print(f"❌ ERROR: Image file not found: {image_path}")
            return None, None
        except Exception as e:
            print(f"❌ ERROR loading image: {str(e)}")
            return None, None
    
    def predict(self):
        """
        Make prediction on loaded image.
        
        Returns:
            tuple: (prediction_label, confidence)
        """
        print(f"\n[STEP 4/4] Making prediction...")
        
        if self.image is None:
            print(f"❌ ERROR: No image loaded. Load an image first.")
            return None, None
        
        try:
            # Prepare image for prediction
            img_input = np.expand_dims(self.image, axis=0)
            
            print(f"   Input shape for model: {img_input.shape}")
            
            # Make prediction
            prediction_prob = self.model.predict(img_input, verbose=0)[0][0]
            
            # Convert to binary label
            prediction_label = 1 if prediction_prob > 0.5 else 0
            confidence = prediction_prob if prediction_label == 1 else (1 - prediction_prob)
            
            self.prediction = prediction_label
            self.confidence = confidence
            
            label_text = "TAMPERED" if prediction_label == 1 else "AUTHENTIC"
            
            print(f"✅ Prediction completed!")
            print(f"   Prediction: {label_text}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"   Raw probability: {prediction_prob:.6f}")
            
            return label_text, confidence
            
        except Exception as e:
            print(f"❌ ERROR during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def display_result_matplotlib(self):
        """
        Display image with prediction result using matplotlib.
        """
        print(f"\n   Displaying result...")
        
        try:
            if self.image_original is None or self.prediction is None:
                print(f"❌ ERROR: No image or prediction to display")
                return
            
            # Prepare label and color
            label_text = "TAMPERED" if self.prediction == 1 else "AUTHENTIC"
            color = (255, 0, 0) if self.prediction == 1 else (0, 255, 0)  # Red/Green
            color_rgb = color[::-1]  # Convert BGR to RGB
            
            # Create figure with result information
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Display image
            ax.imshow(self.image_original)
            ax.axis('off')
            
            # Add title with prediction
            title_text = f"Prediction: {label_text}\nConfidence: {self.confidence*100:.2f}%"
            ax.set_title(title_text, fontsize=16, fontweight='bold', color=color_rgb, pad=20)
            
            # Add filename
            filename = os.path.basename(self.image_path)
            fig.text(0.5, 0.02, f"File: {filename}", ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.show()
            
            print(f"✅ Result displayed!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not display result: {str(e)}")
    
    def display_result_cv2(self):
        """
        Display image with prediction result using OpenCV.
        """
        print(f"\n   Displaying result (OpenCV)...")
        
        try:
            if self.image_original is None or self.prediction is None:
                print(f"❌ ERROR: No image or prediction to display")
                return
            
            # Prepare label and color
            label_text = "TAMPERED" if self.prediction == 1 else "AUTHENTIC"
            color = (0, 0, 255) if self.prediction == 1 else (0, 255, 0)  # Red/Green
            
            # Create a copy for annotation
            display_image = self.image_original.copy()
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            
            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            # Prediction label
            text = f"{label_text}"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x = (display_image.shape[1] - text_size[0]) // 2
            y = 50
            cv2.rectangle(display_image, (x-10, y-30), (x+text_size[0]+10, y+10), color, -1)
            cv2.putText(display_image, text, (x, y), font, font_scale, (255, 255, 255), thickness)
            
            # Confidence
            confidence_text = f"Confidence: {self.confidence*100:.2f}%"
            text_size = cv2.getTextSize(confidence_text, font, 1, 2)[0]
            x = (display_image.shape[1] - text_size[0]) // 2
            y = display_image.shape[0] - 30
            cv2.putText(display_image, confidence_text, (x, y), font, 1, (255, 255, 255), 2)
            
            # Display in window
            cv2.namedWindow('Forgery Detection Result', cv2.WINDOW_NORMAL)
            cv2.imshow('Forgery Detection Result', display_image)
            
            print(f"✅ Result displayed!")
            print(f"   Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"⚠️ Warning: Could not display result: {str(e)}")
    
    def print_detailed_report(self):
        """
        Print detailed prediction report.
        """
        print(f"\n{'='*70}")
        print(f"DETAILED PREDICTION REPORT")
        print(f"{'='*70}")
        print(f"Image Path:     {self.image_path}")
        print(f"File Size:      {os.path.getsize(self.image_path) / 1024:.2f} KB")
        print(f"Image Shape:    {self.image_original.shape}")
        print(f"Prediction:     {'TAMPERED' if self.prediction == 1 else 'AUTHENTIC'}")
        print(f"Confidence:     {self.confidence*100:.2f}%")
        print(f"Probability:    {self.confidence:.6f}")
        print(f"{'='*70}\n")
    
    def test_image(self, image_path=None):
        """
        Complete testing pipeline for a single image.
        
        Args:
            image_path (str): Path to image (optional, uses dialog if not provided)
            
        Returns:
            dict: Prediction result dictionary
        """
        try:
            # Select image if not provided
            if image_path is None:
                image_path = self.select_image_gui()
            
            if image_path is None:
                print(f"⚠️ No image selected. Exiting.")
                return None
            
            # Load image
            processed_img, original_img = self.load_image_file(image_path)
            if processed_img is None:
                print(f"❌ Failed to load image")
                return None
            
            # Make prediction
            label, confidence = self.predict()
            if label is None:
                print(f"❌ Failed to make prediction")
                return None
            
            # Print detailed report
            self.print_detailed_report()
            
            # Display result
            self.display_result_cv2()
            
            # Return result dictionary
            result = {
                'image_path': self.image_path,
                'prediction': label,
                'confidence': float(confidence),
                'probability': float(self.confidence)
            }
            
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR during testing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_test_directory(self, directory_path):
        """
        Test all images in a directory.
        
        Args:
            directory_path (str): Path to directory containing images
            
        Returns:
            list: List of prediction results
        """
        print(f"\n[BATCH TEST] Testing all images in: {directory_path}")
        
        if not os.path.isdir(directory_path):
            print(f"❌ ERROR: Directory not found: {directory_path}")
            return []
        
        results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG')
        
        try:
            image_files = [f for f in os.listdir(directory_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not image_files:
                print(f"⚠️ No image files found in {directory_path}")
                return []
            
            print(f"Found {len(image_files)} images to process\n")
            
            for idx, img_file in enumerate(image_files, 1):
                print(f"[{idx}/{len(image_files)}] Processing: {img_file}")
                
                img_path = os.path.join(directory_path, img_file)
                result = self.test_image(img_path)
                
                if result:
                    results.append(result)
                    print(f"  ✅ {result['prediction']} (Confidence: {result['confidence']:.2%})\n")
                else:
                    print(f"  ❌ Failed to process\n")
            
            # Summary
            print(f"\n{'='*70}")
            print(f"BATCH TEST SUMMARY")
            print(f"{'='*70}")
            print(f"Total processed:  {len(results)}")
            tampered_count = sum(1 for r in results if r['prediction'] == 'TAMPERED')
            authentic_count = len(results) - tampered_count
            print(f"Tampered images:  {tampered_count}")
            print(f"Authentic images: {authentic_count}")
            print(f"{'='*70}\n")
            
            return results
            
        except Exception as e:
            print(f"❌ ERROR during batch testing: {str(e)}")
            return results


def main():
    """
    Main testing function.
    Interactive testing interface.
    """
    print("\n" + "="*70)
    print("IMAGE FORGERY DETECTION - TESTING INTERFACE")
    print("="*70)
    
    try:
        # Get model path
        model_path = '../models/forgery_model.h5'
        
        # Initialize tester
        tester = SingleImageTester(model_path)
        
        # Interactive menu
        while True:
            print(f"\n{'='*70}")
            print(f"MENU OPTIONS")
            print(f"{'='*70}")
            print(f"1. Test single image (GUI)")
            print(f"2. Test single image (specify path)")
            print(f"3. Batch test directory")
            print(f"4. Exit")
            print(f"{'='*70}")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                print(f"\n{'='*70}")
                print(f"TEST SINGLE IMAGE - GUI MODE")
                print(f"{'='*70}")
                result = tester.test_image()
                
                if result:
                    print(f"\n✅ TEST COMPLETED!")
                    print(f"   Prediction: {result['prediction']}")
                    print(f"   Confidence: {result['confidence']:.2%}")
                
            elif choice == '2':
                print(f"\n{'='*70}")
                print(f"TEST SINGLE IMAGE - PATH MODE")
                print(f"{'='*70}")
                img_path = input("Enter image path: ").strip()
                
                if os.path.exists(img_path):
                    result = tester.test_image(img_path)
                    if result:
                        print(f"\n✅ TEST COMPLETED!")
                        print(f"   Prediction: {result['prediction']}")
                        print(f"   Confidence: {result['confidence']:.2%}")
                else:
                    print(f"❌ ERROR: File not found: {img_path}")
            
            elif choice == '3':
                print(f"\n{'='*70}")
                print(f"BATCH TEST - DIRECTORY MODE")
                print(f"{'='*70}")
                dir_path = input("Enter directory path: ").strip()
                
                if os.path.isdir(dir_path):
                    results = tester.batch_test_directory(dir_path)
                    print(f"\n✅ BATCH TEST COMPLETED!")
                    print(f"   Processed {len(results)} images")
                else:
                    print(f"❌ ERROR: Directory not found: {dir_path}")
            
            elif choice == '4':
                print(f"\n✅ Exiting...")
                break
            
            else:
                print(f"❌ Invalid choice. Please enter 1-4.")
    
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print(f"\nSolution: Train the model first using:")
        print(f"  python train_clean_model.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Testing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()