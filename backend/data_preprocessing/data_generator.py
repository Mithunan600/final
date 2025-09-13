import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os

class PlantDiseaseDataGenerator(Sequence):
    """
    Memory-efficient data generator for plant disease images
    """
    
    def __init__(self, data_loader, indices, labels, batch_size=32, img_size=(224, 224), shuffle=True):
        """
        Initialize the data generator
        
        Args:
            data_loader: MemoryEfficientDataLoader instance
            indices: List of indices to use
            labels: Corresponding labels
            batch_size: Batch size
            img_size: Image size for preprocessing
            shuffle: Whether to shuffle data
        """
        self.data_loader = data_loader
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        self.on_epoch_end()
    
    def __len__(self):
        """
        Number of batches per epoch
        """
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_images = np.zeros((len(batch_indices), *self.img_size, 3), dtype=np.float32)
        batch_labels_array = np.array(batch_labels)
        
        # Load and preprocess images
        for i, idx in enumerate(batch_indices):
            # Get the image path and load it
            img_path = self.data_loader.image_paths[idx]
            image = self.data_loader.preprocess_image(img_path)
            
            if image is not None:
                batch_images[i] = image
            else:
                # Fallback: create a random image if loading fails
                batch_images[i] = np.random.random((*self.img_size, 3)).astype(np.float32)
        
        return batch_images, batch_labels_array
    
    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        if self.shuffle:
            # Shuffle indices and labels together
            combined = list(zip(self.indices, self.labels))
            np.random.shuffle(combined)
            self.indices, self.labels = zip(*combined)
            self.indices = list(self.indices)
            self.labels = list(self.labels)

class MemoryEfficientDataLoader:
    """
    Memory-efficient version of PlantDiseaseDataLoader
    """
    
    def __init__(self, data_path, img_size=(224, 224), max_samples_per_class=None):
        """
        Initialize the memory-efficient data loader
        
        Args:
            data_path (str): Path to the dataset folder
            img_size (tuple): Target image size for preprocessing
            max_samples_per_class (int): Maximum samples per class (for testing)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.max_samples_per_class = max_samples_per_class
        
        # Store file paths and labels instead of loading all images
        self.image_paths = []
        self.labels = []
        self.label_encoder = None
        
    def load_plantvillage_data(self):
        """
        Load PlantVillage dataset file paths (not images)
        """
        print("Loading PlantVillage dataset paths...")
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder}")
                
                # Extract plant type and disease from folder name
                if "___" in folder:
                    parts = folder.split("___")
                    plant_type = parts[0]
                    disease = parts[1] if len(parts) > 1 else "healthy"
                    label = f"{plant_type}_{disease}"
                else:
                    label = folder
                
                # Get image files from this folder
                image_files = []
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(folder_path, img_file))
                
                # Limit samples per class if specified
                if self.max_samples_per_class and len(image_files) > self.max_samples_per_class:
                    image_files = image_files[:self.max_samples_per_class]
                
                # Add to our lists
                self.image_paths.extend(image_files)
                self.labels.extend([label] * len(image_files))
        
        print(f"Loaded {len(self.image_paths)} image paths with {len(set(self.labels))} classes")
        return self.image_paths, self.labels
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing indices
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Split indices
        indices = np.arange(len(self.labels))
        train_indices, test_indices, y_train, y_test = train_test_split(
            indices, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"Training set: {len(train_indices)} images")
        print(f"Testing set: {len(test_indices)} images")
        
        return train_indices, test_indices, y_train, y_test
    
    def get_class_names(self):
        """
        Get the class names
        """
        if self.label_encoder:
            return self.label_encoder.classes_.tolist()
        return list(set(self.labels))
    
    def preprocess_image(self, img_path):
        """
        Preprocess a single image from path
        """
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}")
            return None
    
    def get_generator(self, indices, labels, batch_size=32, shuffle=True):
        """
        Get a data generator for the given indices and labels
        """
        return PlantDiseaseDataGenerator(
            self, indices, labels, batch_size, self.img_size, shuffle
        )

# Example usage
if __name__ == "__main__":
    # Test the memory-efficient loader
    loader = MemoryEfficientDataLoader(
        data_path="../datasets/plantvillage/color",
        max_samples_per_class=100  # Limit for testing
    )
    
    # Load data paths
    loader.load_plantvillage_data()
    
    # Create train-test split
    train_indices, test_indices, y_train, y_test = loader.create_train_test_split()
    
    print("Memory-efficient data loader test completed!")
