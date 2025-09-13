import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class PlantDiseaseDataLoader:
    def __init__(self, data_path, img_size=(224, 224)):
        """
        Initialize the data loader for plant disease datasets
        
        Args:
            data_path (str): Path to the dataset folder
            img_size (tuple): Target image size for preprocessing
        """
        self.data_path = data_path
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
    def load_plantvillage_data(self):
        """
        Load PlantVillage dataset which is organized by plant type and disease
        Expected structure:
        dataset/
        ├── Tomato___Bacterial_spot/
        ├── Tomato___Early_blight/
        ├── Tomato___Late_blight/
        ├── Tomato___healthy/
        └── ...
        """
        print("Loading PlantVillage dataset...")
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder}")
                
                # Extract plant type and disease from folder name
                # Format: "Plant___Disease" or "Plant___healthy"
                if "___" in folder:
                    parts = folder.split("___")
                    plant_type = parts[0]
                    disease = parts[1] if len(parts) > 1 else "healthy"
                    label = f"{plant_type}_{disease}"
                else:
                    label = folder
                
                # Load images from this folder
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, img_file)
                        
                        try:
                            # Load and preprocess image
                            image = self.preprocess_image(img_path)
                            if image is not None:
                                self.images.append(image)
                                self.labels.append(label)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            continue
        
        print(f"Loaded {len(self.images)} images with {len(set(self.labels))} classes")
        return self.images, self.labels
    
    def load_plantdoc_data(self):
        """
        Load PlantDoc dataset
        """
        print("Loading PlantDoc dataset...")
        
        # PlantDoc structure might be different, adjust as needed
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    
                    # Extract label from folder structure
                    # Adjust this based on actual PlantDoc structure
                    relative_path = os.path.relpath(root, self.data_path)
                    label = relative_path.replace(os.sep, "_")
                    
                    try:
                        image = self.preprocess_image(img_path)
                        if image is not None:
                            self.images.append(image)
                            self.labels.append(label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        
        print(f"Loaded {len(self.images)} images with {len(set(self.labels))} classes")
        return self.images, self.labels
    
    def preprocess_image(self, img_path):
        """
        Preprocess a single image
        
        Args:
            img_path (str): Path to the image
            
        Returns:
            numpy.ndarray: Preprocessed image array
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
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # For large datasets, we'll use indices instead of loading all images
        from sklearn.model_selection import train_test_split
        
        # Create indices
        indices = np.arange(len(self.labels))
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Split indices
        train_indices, test_indices, y_train, y_test = train_test_split(
            indices, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Store indices for later use
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        print(f"Training set: {len(train_indices)} images")
        print(f"Testing set: {len(test_indices)} images")
        
        return train_indices, test_indices, y_train, y_test
    
    def get_class_names(self):
        """
        Get the class names (plant types and diseases)
        
        Returns:
            list: List of class names
        """
        return self.label_encoder.classes_.tolist()
    
    def plot_class_distribution(self, save_path=None):
        """
        Plot the distribution of classes in the dataset
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        # Count occurrences of each class
        class_counts = pd.Series(self.labels).value_counts()
        
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(class_counts)), class_counts.values)
        plt.xlabel('Class Index')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Plant Disease Classes')
        plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def display_sample_images(self, num_samples=16, save_path=None):
        """
        Display sample images from the dataset
        
        Args:
            num_samples (int): Number of sample images to display
            save_path (str): Path to save the plot (optional)
        """
        # Get unique indices for different classes
        unique_labels = list(set(self.labels))
        samples_per_class = num_samples // len(unique_labels)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        idx = 0
        for label in unique_labels[:4]:  # Show first 4 classes
            # Find indices for this class
            class_indices = [i for i, l in enumerate(self.labels) if l == label]
            
            for i in range(min(samples_per_class, len(class_indices))):
                if idx < 16:
                    img_idx = class_indices[i]
                    axes[idx].imshow(self.images[img_idx])
                    axes[idx].set_title(f"{label}")
                    axes[idx].axis('off')
                    idx += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = PlantDiseaseDataLoader(
        data_path="path/to/your/dataset",  # Update this path
        img_size=(224, 224)
    )
    
    # Load data (choose appropriate method based on your dataset)
    # data_loader.load_plantvillage_data()
    # data_loader.load_plantdoc_data()
    
    # Create train-test split
    # X_train, X_test, y_train, y_test = data_loader.create_train_test_split()
    
    # Get class names
    # class_names = data_loader.get_class_names()
    # print("Classes:", class_names)
