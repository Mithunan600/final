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
from data_preprocessing.data_generator import MemoryEfficientDataLoader
from model_architecture import PlantDiseaseModel, DataAugmentation

class MultiDatasetTrainer:
    def __init__(self, datasets_config, model_save_path="models/", results_path="results/"):
        """
        Initialize the multi-dataset trainer
        
        Args:
            datasets_config (dict): Configuration for multiple datasets
            model_save_path (str): Path to save trained models
            results_path (str): Path to save training results
        """
        self.datasets_config = datasets_config
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
        
        # Combined data
        self.combined_image_paths = []
        self.combined_labels = []
        self.combined_class_names = []
        
    def load_multiple_datasets(self):
        """
        Load and combine multiple datasets
        """
        print("🔄 Loading multiple datasets...")
        
        all_image_paths = []
        all_labels = []
        all_class_names = set()
        
        for dataset_name, config in self.datasets_config.items():
            print(f"\n📁 Loading {dataset_name}...")
            
            # Initialize data loader for this dataset
            loader = MemoryEfficientDataLoader(
                data_path=config["path"],
                img_size=(224, 224),
                max_samples_per_class=config.get("max_samples_per_class", None)
            )
            
            # Load dataset
            if config["type"] == "plantvillage":
                loader.load_plantvillage_data()
            elif config["type"] == "rice":
                # You can implement rice dataset loading here
                print(f"⚠️  Rice dataset loading not implemented yet for {dataset_name}")
                continue
            else:
                print(f"⚠️  Unknown dataset type: {config['type']}")
                continue
            
            # Add prefix to labels to avoid conflicts
            prefix = config.get("prefix", dataset_name)
            prefixed_labels = [f"{prefix}_{label}" for label in loader.labels]
            
            # Combine data
            all_image_paths.extend(loader.image_paths)
            all_labels.extend(prefixed_labels)
            all_class_names.update(prefixed_labels)
            
            print(f"   ✅ Loaded {len(loader.image_paths)} images with {len(set(prefixed_labels))} classes")
        
        # Store combined data
        self.combined_image_paths = all_image_paths
        self.combined_labels = all_labels
        self.combined_class_names = sorted(list(all_class_names))
        
        print(f"\n🎯 Combined dataset summary:")
        print(f"   Total images: {len(self.combined_image_paths)}")
        print(f"   Total classes: {len(self.combined_class_names)}")
        print(f"   Classes: {self.combined_class_names[:10]}...")  # Show first 10 classes
        
        # Create a new data loader with combined data
        self.data_loader = MemoryEfficientDataLoader(
            data_path="",  # We'll set the data manually
            img_size=(224, 224)
        )
        
        # Set the combined data
        self.data_loader.image_paths = self.combined_image_paths
        self.data_loader.labels = self.combined_labels
        
        # Create label encoder
        from sklearn.preprocessing import LabelEncoder
        self.data_loader.label_encoder = LabelEncoder()
        self.data_loader.label_encoder.fit(self.combined_labels)
        
        return self.data_loader
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create train-test split from combined dataset
        """
        from sklearn.model_selection import train_test_split
        
        # Encode labels
        y_encoded = self.data_loader.label_encoder.transform(self.combined_labels)
        
        # Split indices
        indices = np.arange(len(self.combined_labels))
        train_indices, test_indices, y_train, y_test = train_test_split(
            indices, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"📊 Train-test split:")
        print(f"   Training samples: {len(train_indices)}")
        print(f"   Testing samples: {len(test_indices)}")
        
        return train_indices, test_indices, y_train, y_test
    
    def create_model(self, model_type="transfer_learning", base_model="mobilenet"):
        """
        Create the model architecture
        """
        print(f"🏗️  Creating {model_type} model...")
        
        # Initialize model builder
        self.model_builder = PlantDiseaseModel(
            num_classes=len(self.combined_class_names),
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
        
        print("✅ Model created and compiled successfully!")
        print(f"   Model summary:")
        self.model.summary()
    
    def train_model(self, train_indices, y_train, test_indices, y_test, epochs=50, batch_size=32):
        """
        Train the model using combined dataset
        """
        print("🚀 Starting model training...")
        
        # Get callbacks
        model_save_path = os.path.join(self.model_save_path, 'multi_dataset_model.keras')
        callbacks = self.model_builder.get_callbacks(model_save_path)
        
        # Create data generators
        train_generator = self.data_loader.get_generator(
            train_indices, y_train, batch_size=batch_size, shuffle=True
        )
        
        test_generator = self.data_loader.get_generator(
            test_indices, y_test, batch_size=batch_size, shuffle=False
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
        
        print("✅ Training completed!")
    
    def evaluate_model(self, test_indices, y_test):
        """
        Evaluate the trained model
        """
        print("📊 Evaluating model...")
        
        # Create test data generator
        test_generator = self.data_loader.get_generator(
            test_indices, y_test, batch_size=32, shuffle=False
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"🎯 Test Loss: {test_loss:.4f}")
        
        # Make predictions
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Ensure y_test and y_pred have the same length
        min_length = min(len(y_test), len(y_pred))
        y_test = y_test[:min_length]
        y_pred = y_pred[:min_length]
        
        # Generate classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.combined_class_names,
            output_dict=True
        )
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.combined_class_names))
        
        # Save results
        with open(os.path.join(self.results_path, 'multi_dataset_results.json'), 'w') as f:
            json.dump({
                "accuracy": float(test_accuracy),
                "loss": float(test_loss),
                "num_classes": len(self.combined_class_names),
                "class_names": self.combined_class_names,
                "classification_report": report
            }, f, indent=2)
        
        return test_accuracy, test_loss, report
    
    def save_model_info(self):
        """
        Save model and dataset information
        """
        # Save class names
        with open(os.path.join(self.results_path, 'multi_dataset_class_names.json'), 'w') as f:
            json.dump(self.combined_class_names, f, indent=2)
        
        # Save label encoder
        with open(os.path.join(self.results_path, 'multi_dataset_label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.data_loader.label_encoder, f)
        
        print("💾 Model information saved!")

def main():
    """
    Main function to train model with multiple datasets
    """
    # Configuration for multiple datasets
    datasets_config = {
        "plantvillage": {
            "path": "../datasets/plantvillage/color",
            "type": "plantvillage",
            "max_samples_per_class": 1000,  # Limit for memory efficiency
            "prefix": "PV"  # Prefix to avoid label conflicts
        },
        # Add more datasets here as you download them
        # "rice_diseases": {
        #     "path": "../datasets/rice_diseases",
        #     "type": "rice",
        #     "max_samples_per_class": 500,
        #     "prefix": "RD"
        # },
        # "new_plant_diseases": {
        #     "path": "../datasets/new_plant_diseases",
        #     "type": "plantvillage",
        #     "max_samples_per_class": 800,
        #     "prefix": "NPD"
        # }
    }
    
    # Initialize trainer
    trainer = MultiDatasetTrainer(
        datasets_config=datasets_config,
        model_save_path="models/",
        results_path="results/"
    )
    
    # Load multiple datasets
    trainer.load_multiple_datasets()
    
    # Create train-test split
    train_indices, test_indices, y_train, y_test = trainer.create_train_test_split()
    
    # Create model
    trainer.create_model(model_type="transfer_learning", base_model="mobilenet")
    
    # Train model
    trainer.train_model(train_indices, y_train, test_indices, y_test, epochs=10)  # Reduced for testing
    
    # Evaluate model
    accuracy, loss, report = trainer.evaluate_model(test_indices, y_test)
    
    # Save model info
    trainer.save_model_info()
    
    print(f"\n🎉 Multi-dataset training completed!")
    print(f"📊 Final Accuracy: {accuracy:.4f}")
    print(f"💾 Model saved to: {trainer.model_save_path}")
    print(f"📋 Results saved to: {trainer.results_path}")

if __name__ == "__main__":
    main()
