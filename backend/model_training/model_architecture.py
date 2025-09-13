import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class PlantDiseaseModel:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        """
        Initialize the plant disease classification model
        
        Args:
            num_classes (int): Number of disease classes
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def create_custom_cnn(self):
        """
        Create a custom CNN model for plant disease classification
        
        Returns:
            keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self, base_model='mobilenet'):
        """
        Create a transfer learning model using pre-trained architectures
        
        Args:
            base_model (str): Base model architecture ('mobilenet', 'efficientnet', 'resnet')
            
        Returns:
            keras.Model: Compiled transfer learning model
        """
        # Select base model
        if base_model == 'mobilenet':
            base = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_model == 'efficientnet':
            base = EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_model == 'resnet':
            base = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError("Supported base models: 'mobilenet', 'efficientnet', 'resnet'")
        
        # Freeze base model layers initially
        base.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled model
        """
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_save_path, patience=10):
        """
        Get training callbacks
        
        Args:
            model_save_path (str): Path to save the best model
            patience (int): Patience for early stopping
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            ModelCheckpoint(
                filepath=model_save_path.replace('.h5', '.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def fine_tune_model(self, model, learning_rate=1e-5):
        """
        Fine-tune the transfer learning model by unfreezing some layers
        
        Args:
            model: Pre-trained model
            learning_rate (float): Lower learning rate for fine-tuning
            
        Returns:
            keras.Model: Fine-tuned model
        """
        # Unfreeze the top layers of the base model
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 30
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class DataAugmentation:
    """
    Data augmentation class for plant disease images
    """
    
    @staticmethod
    def get_training_augmentation():
        """
        Get data augmentation for training
        
        Returns:
            keras.Sequential: Data augmentation pipeline
        """
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ])
    
    @staticmethod
    def get_validation_preprocessing():
        """
        Get preprocessing for validation (no augmentation)
        
        Returns:
            keras.Sequential: Preprocessing pipeline
        """
        return keras.Sequential([
            # Only normalization, no augmentation
        ])

# Example usage
if __name__ == "__main__":
    # Example: Create a model for 38 classes (PlantVillage dataset)
    model_builder = PlantDiseaseModel(num_classes=38)
    
    # Create transfer learning model
    model = model_builder.create_transfer_learning_model(base_model='mobilenet')
    
    # Compile model
    model = model_builder.compile_model(model)
    
    # Print model summary
    model.summary()
