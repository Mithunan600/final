import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import pickle

# Import our custom modules
import sys
sys.path.append('..')
from data_preprocessing.data_loader import PlantDiseaseDataLoader
from data_preprocessing.data_generator import MemoryEfficientDataLoader
from model_architecture import PlantDiseaseModel, DataAugmentation

class ModelTrainer:
    def __init__(self, data_path, model_save_path="models/", results_path="results/"):
        """
        Initialize the model trainer
        
        Args:
            data_path (str): Path to the dataset
            model_save_path (str): Path to save trained models
            results_path (str): Path to save training results
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.results_path = results_path
        
        # Create directories if they don't exist
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.model_builder = None
        self.model = None
        self.history = None
        
    def load_and_prepare_data(self, dataset_type="plantvillage"):
        """
        Load and prepare the dataset using memory-efficient approach
        
        Args:
            dataset_type (str): Type of dataset ('plantvillage' or 'plantdoc')
        """
        print("Loading and preparing data...")
        
        # Initialize memory-efficient data loader
        self.data_loader = MemoryEfficientDataLoader(
            data_path=self.data_path,
            img_size=(224, 224),
            max_samples_per_class=1000  # Limit for memory efficiency
        )
        
        # Load data based on dataset type
        if dataset_type == "plantvillage":
            self.data_loader.load_plantvillage_data()
        elif dataset_type == "plantdoc":
            # For PlantDoc, we'd need to implement similar method
            raise NotImplementedError("PlantDoc support not implemented yet")
        else:
            raise ValueError("Supported dataset types: 'plantvillage'")
        
        # Create train-test split
        self.train_indices, self.test_indices, self.y_train, self.y_test = self.data_loader.create_train_test_split(
            test_size=0.2, random_state=42
        )
        
        # Get class names
        self.class_names = self.data_loader.get_class_names()
        self.num_classes = len(self.class_names)
        
        print(f"Dataset prepared:")
        print(f"  Training samples: {len(self.train_indices)}")
        print(f"  Testing samples: {len(self.test_indices)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes: {self.class_names[:5]}...")  # Show first 5 classes
        
        # Save class names for later use
        with open(os.path.join(self.results_path, 'class_names.json'), 'w') as f:
            json.dump(self.class_names, f)
        
        # Save label encoder
        with open(os.path.join(self.results_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.data_loader.label_encoder, f)
    
    def create_model(self, model_type="transfer_learning", base_model="mobilenet"):
        """
        Create the model architecture
        
        Args:
            model_type (str): Type of model ('custom_cnn' or 'transfer_learning')
            base_model (str): Base model for transfer learning ('mobilenet', 'efficientnet', 'resnet')
        """
        print(f"Creating {model_type} model...")
        
        # Initialize model builder
        self.model_builder = PlantDiseaseModel(
            num_classes=self.num_classes,
            input_shape=(224, 224, 3)
        )
        
        # Create model based on type
        if model_type == "custom_cnn":
            self.model = self.model_builder.create_custom_cnn()
        elif model_type == "transfer_learning":
            self.model = self.model_builder.create_transfer_learning_model(base_model=base_model)
        else:
            raise ValueError("Supported model types: 'custom_cnn', 'transfer_learning'")
        
        # Compile model
        self.model = self.model_builder.compile_model(self.model)
        
        print("Model created and compiled successfully!")
        print(f"Model summary:")
        self.model.summary()
    
    def train_model(self, epochs=50, batch_size=32, use_augmentation=True):
        """
        Train the model using memory-efficient data generators
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            use_augmentation (bool): Whether to use data augmentation
        """
        print("Starting model training...")
        
        # Get callbacks
        model_save_path = os.path.join(self.model_save_path, 'best_model.keras')
        callbacks = self.model_builder.get_callbacks(model_save_path)
        
        # Create data generators
        train_generator = self.data_loader.get_generator(
            self.train_indices, self.y_train, batch_size=batch_size, shuffle=True
        )
        
        test_generator = self.data_loader.get_generator(
            self.test_indices, self.y_test, batch_size=batch_size, shuffle=False
        )
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=len(test_generator),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
    
    def fine_tune_model(self, epochs=20, batch_size=16):
        """
        Fine-tune the transfer learning model
        
        Args:
            epochs (int): Number of fine-tuning epochs
            batch_size (int): Batch size for fine-tuning
        """
        print("Starting model fine-tuning...")
        
        # Fine-tune the model
        self.model = self.model_builder.fine_tune_model(self.model)
        
        # Get callbacks for fine-tuning
        model_save_path = os.path.join(self.model_save_path, 'fine_tuned_model.h5')
        callbacks = self.model_builder.get_callbacks(model_save_path)
        
        # Create data generators for fine-tuning
        train_generator = self.data_loader.get_generator(
            self.train_indices, self.y_train, batch_size=batch_size, shuffle=True
        )
        
        test_generator = self.data_loader.get_generator(
            self.test_indices, self.y_test, batch_size=batch_size, shuffle=False
        )
        
        # Fine-tune with lower learning rate
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=len(test_generator),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Fine-tuning completed!")
    
    def evaluate_model(self):
        """
        Evaluate the trained model and generate reports
        """
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate classification report
        report = classification_report(
            self.y_test, y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save classification report
        with open(os.path.join(self.results_path, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy, test_loss, report, cm
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_info(self, model_info):
        """
        Save model information
        
        Args:
            model_info (dict): Dictionary containing model information
        """
        with open(os.path.join(self.results_path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)

# Main training function
def main():
    """
    Main function to run the complete training pipeline
    """
    # Configuration
    config = {
        "data_path": "../datasets/plantvillage/color",  # Updated path to our downloaded dataset
        "dataset_type": "plantvillage",  # or "plantdoc"
        "model_type": "transfer_learning",  # or "custom_cnn"
        "base_model": "mobilenet",  # or "efficientnet", "resnet"
        "epochs": 5,  # Reduced for testing
        "batch_size": 32,
        "use_augmentation": True,
        "fine_tune": True,
        "fine_tune_epochs": 20
    }
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_path=config["data_path"],
        model_save_path="models/",
        results_path="results/"
    )
    
    # Load and prepare data
    trainer.load_and_prepare_data(dataset_type=config["dataset_type"])
    
    # Create model
    trainer.create_model(
        model_type=config["model_type"],
        base_model=config["base_model"]
    )
    
    # Train model
    trainer.train_model(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        use_augmentation=config["use_augmentation"]
    )
    
    # Fine-tune if specified
    if config["fine_tune"]:
        trainer.fine_tune_model(
            epochs=config["fine_tune_epochs"],
            batch_size=config["batch_size"] // 2
        )
    
    # Evaluate model
    accuracy, loss, report, cm = trainer.evaluate_model()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model information
    model_info = {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "num_classes": trainer.num_classes,
        "class_names": trainer.class_names,
        "config": config
    }
    trainer.save_model_info(model_info)
    
    print("Training pipeline completed successfully!")
    print(f"Model saved to: {trainer.model_save_path}")
    print(f"Results saved to: {trainer.results_path}")

if __name__ == "__main__":
    main()
