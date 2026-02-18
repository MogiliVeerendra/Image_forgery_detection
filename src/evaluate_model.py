"""
Image Forgery Detection - Model Evaluation Module
================================================
Comprehensive evaluation of trained CNN model on test dataset.
Generates metrics, confusion matrix, ROC curve, and detailed reports.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_image
from preprocess_and_load import DatasetLoader


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    Handles model loading, prediction, and metric calculation.
    """
    
    def __init__(self, model_path='../models/forgery_model.h5', 
                 dataset_path='../datasets', 
                 test_split=0.2):
        """
        Initialize ModelEvaluator.
        
        Args:
            model_path (str): Path to trained .h5 model
            dataset_path (str): Path to datasets directory
            test_split (float): Test set proportion
        """
        print("\n" + "="*70)
        print("INITIALIZING MODEL EVALUATOR")
        print("="*70)
        
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.test_split = test_split
        self.model = None
        self.test_images = None
        self.test_labels = None
        self.predictions = None
        self.predictions_proba = None
        
        # Create models directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """
        Load trained model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        print(f"\n[STEP 1/5] Loading model from: {self.model_path}")
        
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
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR loading model: {str(e)}")
            sys.exit(1)
    
    def prepare_test_data(self):
        """
        Load and prepare test dataset.
        
        Returns:
            tuple: (test_images, test_labels)
        """
        print(f"\n[STEP 2/5] Preparing test dataset...")
        
        try:
            # Use DatasetLoader to get test data
            loader = DatasetLoader(self.dataset_path)
            
            # load_all_datasets() returns (X_test, y_test, class_weights)
            result = loader.load_all_datasets()
            
            if len(result) == 3:
                # Returns X_test, y_test, class_weights
                X_test, y_test, _ = result
            else:
                # Fallback for different return format
                X_test, y_test = result[0], result[1]
            
            self.test_images = X_test
            self.test_labels = y_test
            
            print(f"✅ Test dataset prepared!")
            print(f"   Test images shape: {X_test.shape}")
            print(f"   Test labels shape: {y_test.shape}")
            print(f"   Authentic (0): {np.sum(y_test == 0)}")
            print(f"   Tampered (1): {np.sum(y_test == 1)}")
            
            return X_test, y_test
            
        except Exception as e:
            print(f"❌ ERROR preparing test data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def make_predictions(self):
        """
        Generate predictions on test set.
        
        Returns:
            tuple: (predictions_binary, predictions_probability)
        """
        print(f"\n[STEP 3/5] Making predictions on test set...")
        
        if self.test_images is None:
            print("❌ ERROR: Test data not prepared. Call prepare_test_data() first.")
            return None, None
        
        try:
            print(f"   Predicting on {len(self.test_images)} images...")
            
            # Get probability predictions
            self.predictions_proba = self.model.predict(
                self.test_images,
                batch_size=32,
                verbose=0
            )
            
            # Convert to binary predictions (threshold = 0.5)
            self.predictions = (self.predictions_proba > 0.5).astype(int).flatten()
            
            print(f"✅ Predictions completed!")
            print(f"   Predictions shape: {self.predictions.shape}")
            print(f"   Probability range: [{self.predictions_proba.min():.4f}, {self.predictions_proba.max():.4f}]")
            
            return self.predictions, self.predictions_proba
            
        except Exception as e:
            print(f"❌ ERROR during prediction: {str(e)}")
            raise
    
    def calculate_metrics(self):
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            dict: Dictionary containing all metrics
        """
        print(f"\n[STEP 4/5] Calculating evaluation metrics...")
        
        if self.predictions is None:
            print("❌ ERROR: Predictions not made. Call make_predictions() first.")
            return None
        
        try:
            # Binary metrics
            accuracy = accuracy_score(self.test_labels, self.predictions)
            precision = precision_score(self.test_labels, self.predictions, zero_division=0)
            recall = recall_score(self.test_labels, self.predictions, zero_division=0)
            f1 = f1_score(self.test_labels, self.predictions, zero_division=0)
            
            # ROC and AUC
            fpr, tpr, _ = roc_curve(self.test_labels, self.predictions_proba)
            roc_auc = auc(fpr, tpr)
            
            # Confusion Matrix
            cm = confusion_matrix(self.test_labels, self.predictions)
            
            # Classification Report
            class_report = classification_report(
                self.test_labels, 
                self.predictions,
                target_names=['Authentic', 'Tampered'],
                output_dict=True
            )
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report
            }
            
            # Print metrics
            print(f"\n{'='*70}")
            print(f"EVALUATION METRICS")
            print(f"{'='*70}")
            print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"{'='*70}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ ERROR calculating metrics: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, metrics, output_path='../models/confusion_matrix.png'):
        """
        Plot and save confusion matrix heatmap.
        
        Args:
            metrics (dict): Metrics dictionary containing confusion matrix
            output_path (str): Path to save plot
        """
        print(f"\n   Generating confusion matrix plot...")
        
        try:
            cm = np.array(metrics['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Authentic', 'Tampered'],
                yticklabels=['Authentic', 'Tampered'],
                cbar_kws={'label': 'Count'}
            )
            plt.title('Confusion Matrix - Forgery Detection Model', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Confusion matrix saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not save confusion matrix: {str(e)}")
    
    def plot_roc_curve(self, metrics, output_path='../models/roc_curve.png'):
        """
        Plot and save ROC curve.
        
        Args:
            metrics (dict): Metrics dictionary containing FPR, TPR, AUC
            output_path (str): Path to save plot
        """
        print(f"   Generating ROC curve...")
        
        try:
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            roc_auc = metrics['roc_auc']
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve - Forgery Detection Model', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ ROC curve saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not save ROC curve: {str(e)}")
    
    def save_metrics(self, metrics, output_path='../models/evaluation_metrics.json'):
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics (dict): Metrics dictionary
            output_path (str): Path to save JSON file
        """
        print(f"\n[STEP 5/5] Saving evaluation results...")
        print(f"   Saving metrics to: {output_path}")
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"✅ Metrics saved successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save metrics: {str(e)}")
    
    def generate_detailed_report(self, metrics):
        """
        Generate and print detailed evaluation report.
        
        Args:
            metrics (dict): Metrics dictionary
        """
        print(f"\n{'='*70}")
        print(f"DETAILED CLASSIFICATION REPORT")
        print(f"{'='*70}\n")
        
        class_report = metrics['classification_report']
        
        print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
        print("-" * 70)
        
        for class_name in ['Authentic', 'Tampered']:
            if class_name in class_report:
                metrics_cls = class_report[class_name]
                print(f"{class_name:<15} {metrics_cls['precision']:<15.4f} "
                      f"{metrics_cls['recall']:<15.4f} {metrics_cls['f1-score']:<15.4f} "
                      f"{int(metrics_cls['support']):<10}")
        
        print("-" * 70)
        avg_metrics = class_report['weighted avg']
        print(f"{'Weighted Avg':<15} {avg_metrics['precision']:<15.4f} "
              f"{avg_metrics['recall']:<15.4f} {avg_metrics['f1-score']:<15.4f}")
        print(f"{'='*70}\n")
    
    def evaluate(self):
        """
        Complete evaluation pipeline.
        Prepares data, makes predictions, calculates metrics, and generates visualizations.
        
        Returns:
            dict: Evaluation metrics dictionary
        """
        try:
            self.prepare_test_data()
            self.make_predictions()
            metrics = self.calculate_metrics()
            self.plot_confusion_matrix(metrics)
            self.plot_roc_curve(metrics)
            self.generate_detailed_report(metrics)
            self.save_metrics(metrics)
            
            print(f"\n{'='*70}")
            print(f"✅ EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"{'='*70}\n")
            
            return metrics
            
        except Exception as e:
            print(f"\n❌ ERROR during evaluation: {str(e)}")
            raise


def main():
    """
    Main evaluation function.
    Handles model evaluation workflow.
    """
    print("\n" + "="*70)
    print("IMAGE FORGERY DETECTION - MODEL EVALUATION")
    print("="*70)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path='../models/forgery_model.h5')
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        print(f"Evaluation results saved to: ../models/")
        print(f"Generated files:")
        print(f"  ✅ evaluation_metrics.json")
        print(f"  ✅ confusion_matrix.png")
        print(f"  ✅ roc_curve.png")
        
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print(f"\nSolution: Train the model first using:")
        print(f"  python train_clean_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()