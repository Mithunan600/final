import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import pickle
import cv2

# Import our custom modules
import sys
sys.path.append('..')
from data_preprocessing.data_generator import MemoryEfficientDataLoader

class ModelEvaluator:
    def __init__(self, model_path, data_path, results_path="results/"):
        """
        Initialize the model evaluator
        
        Args:
            model_path (str): Path to the trained model
            data_path (str): Path to the dataset
            results_path (str): Path to save evaluation results
        """
        self.model_path = model_path
        self.data_path = data_path
        self.results_path = results_path
        
        # Create results directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)
        
        self.model = None
        self.data_loader = None
        self.class_names = None
        
    def load_model(self):
        """
        Load the trained model
        """
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
            print(f"Model summary:")
            self.model.summary()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_data(self):
        """
        Load the dataset for evaluation
        """
        print("Loading dataset for evaluation...")
        
        # Initialize data loader
        self.data_loader = MemoryEfficientDataLoader(
            data_path=self.data_path,
            img_size=(224, 224)
        )
        
        # Load data paths
        self.data_loader.load_plantvillage_data()
        
        # Create train-test split
        train_indices, test_indices, y_train, y_test = self.data_loader.create_train_test_split(
            test_size=0.2, random_state=42
        )
        
        # Get class names
        self.class_names = self.data_loader.get_class_names()
        
        print(f"Dataset loaded:")
        print(f"  Test samples: {len(test_indices)}")
        print(f"  Number of classes: {len(self.class_names)}")
        
        return test_indices, y_test
    
    def predict_single_image(self, image_path):
        """
        Predict disease for a single image
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            tuple: (predicted_class, confidence, all_predictions)
        """
        if self.model is None:
            print("Model not loaded!")
            return None
        
        # Preprocess image
        image = self.data_loader.preprocess_image(image_path)
        if image is None:
            return None
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_batch, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def evaluate_model(self, test_indices, y_test, batch_size=32):
        """
        Evaluate the model on test data
        
        Args:
            test_indices: Test data indices
            y_test: Test labels
            batch_size: Batch size for evaluation
        """
        print("Evaluating model...")
        
        # Create test data generator
        test_generator = self.data_loader.get_generator(
            test_indices, y_test, batch_size=batch_size, shuffle=False
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            test_generator,
            verbose=1
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Ensure y_test and y_pred have the same length
        min_length = min(len(y_test), len(y_pred))
        y_test = y_test[:min_length]
        y_pred = y_pred[:min_length]
        predictions = predictions[:min_length]
        
        # Generate detailed evaluation
        self.generate_evaluation_report(y_test, y_pred, predictions)
        
        return test_accuracy, test_loss, y_pred, predictions
    
    def generate_evaluation_report(self, y_true, y_pred, predictions):
        """
        Generate comprehensive evaluation report
        """
        print("Generating evaluation report...")
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save classification report
        with open(os.path.join(self.results_path, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Plant Disease Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        print("\nPer-Class Accuracy:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {per_class_accuracy[i]:.4f}")
        
        # Save per-class accuracy
        per_class_dict = {class_name: float(accuracy) 
                         for class_name, accuracy in zip(self.class_names, per_class_accuracy)}
        
        with open(os.path.join(self.results_path, 'per_class_accuracy.json'), 'w') as f:
            json.dump(per_class_dict, f, indent=2)
        
        return report, cm
    
    def test_random_samples(self, test_indices, y_test, num_samples=10):
        """
        Test model on random samples and display results
        """
        print(f"Testing on {num_samples} random samples...")
        
        # Select random samples
        random_indices = np.random.choice(test_indices, size=num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(random_indices):
            # Get image path and true label
            img_path = self.data_loader.image_paths[idx]
            true_label_idx = np.where(test_indices == idx)[0][0]
            true_label = self.class_names[y_test[true_label_idx]]
            
            # Make prediction
            pred_class, confidence, all_preds = self.predict_single_image(img_path)
            
            # Load and display image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            axes[i].imshow(image)
            axes[i].set_title(f"True: {true_label}\nPred: {pred_class}\nConf: {confidence:.3f}")
            axes[i].axis('off')
            
            # Color code based on correctness
            if pred_class == true_label:
                axes[i].set_title(f"✓ True: {true_label}\nPred: {pred_class}\nConf: {confidence:.3f}", 
                                color='green')
            else:
                axes[i].set_title(f"✗ True: {true_label}\nPred: {pred_class}\nConf: {confidence:.3f}", 
                                color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'sample_predictions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_info(self, accuracy, loss):
        """
        Save model evaluation information
        """
        model_info = {
            "model_path": self.model_path,
            "test_accuracy": float(accuracy),
            "test_loss": float(loss),
            "num_classes": len(self.class_names),
            "class_names": self.class_names
        }
        
        with open(os.path.join(self.results_path, 'model_evaluation_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)

def main():
    """
    Main function to evaluate the trained model
    """
    # Configuration
    model_path = "models/best_model.keras"  # Update this path
    data_path = "../datasets/plantvillage/color"
    results_path = "results/"
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        data_path=data_path,
        results_path=results_path
    )
    
    # Load model
    if not evaluator.load_model():
        print("Failed to load model. Please check the model path.")
        return
    
    # Load data
    test_indices, y_test = evaluator.load_data()
    
    # Evaluate model
    accuracy, loss, y_pred, predictions = evaluator.evaluate_model(test_indices, y_test)
    
    # Test on random samples
    evaluator.test_random_samples(test_indices, y_test, num_samples=10)
    
    # Save model info
    evaluator.save_model_info(accuracy, loss)
    
    print(f"\nEvaluation completed!")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
