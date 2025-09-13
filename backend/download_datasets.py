import os
import kagglehub
from pathlib import Path

class DatasetDownloader:
    def __init__(self, base_path="datasets/"):
        """
        Initialize dataset downloader
        
        Args:
            base_path (str): Base path to store downloaded datasets
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def download_plantvillage_dataset(self):
        """
        Download PlantVillage dataset from Kaggle
        """
        print("Downloading PlantVillage dataset...")
        
        try:
            # Download using kagglehub
            path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
            
            # Move to our organized structure
            target_path = self.base_path / "plantvillage"
            if not target_path.exists():
                target_path.mkdir(parents=True)
                
                # Copy files (this is a simplified version - you might need to adjust based on actual structure)
                import shutil
                if os.path.exists(path):
                    # Get the actual dataset folder from the download
                    for item in os.listdir(path):
                        if os.path.isdir(os.path.join(path, item)):
                            dataset_folder = os.path.join(path, item)
                            break
                    else:
                        dataset_folder = path
                    
                    # Copy to our organized location
                    shutil.copytree(dataset_folder, target_path, dirs_exist_ok=True)
                    print(f"PlantVillage dataset saved to: {target_path}")
                else:
                    print(f"Downloaded dataset found at: {path}")
            else:
                print("PlantVillage dataset already exists!")
                
        except Exception as e:
            print(f"Error downloading PlantVillage dataset: {e}")
            
    def download_rice_diseases_dataset(self):
        """
        Download Rice Diseases dataset from Kaggle
        """
        print("Downloading Rice Diseases dataset...")
        
        try:
            path = kagglehub.dataset_download("minhhuy2810/rice-diseases-image-dataset")
            
            target_path = self.base_path / "rice_diseases"
            if not target_path.exists():
                target_path.mkdir(parents=True)
                
                import shutil
                if os.path.exists(path):
                    for item in os.listdir(path):
                        if os.path.isdir(os.path.join(path, item)):
                            dataset_folder = os.path.join(path, item)
                            break
                    else:
                        dataset_folder = path
                    
                    shutil.copytree(dataset_folder, target_path, dirs_exist_ok=True)
                    print(f"Rice Diseases dataset saved to: {target_path}")
                else:
                    print(f"Downloaded dataset found at: {path}")
            else:
                print("Rice Diseases dataset already exists!")
                
        except Exception as e:
            print(f"Error downloading Rice Diseases dataset: {e}")
            
    def download_plant_pathology_2020(self):
        """
        Download Plant Pathology 2020 dataset from Kaggle
        """
        print("Downloading Plant Pathology 2020 dataset...")
        
        try:
            # This one is from a competition, might need different approach
            # You can download it manually from Kaggle or use kaggle API
            target_path = self.base_path / "plant_pathology_2020"
            target_path.mkdir(exist_ok=True)
            
            print("Plant Pathology 2020 dataset - please download manually from:")
            print("https://www.kaggle.com/c/plant-pathology-2020-fgvc7")
            print(f"Save to: {target_path}")
            
        except Exception as e:
            print(f"Error with Plant Pathology 2020 dataset: {e}")
    
    def download_all_datasets(self):
        """
        Download all available datasets
        """
        print("Downloading all plant disease datasets...")
        
        self.download_plantvillage_dataset()
        self.download_rice_diseases_dataset()
        self.download_plant_pathology_2020()
        
        print("Dataset download process completed!")
        
    def list_available_datasets(self):
        """
        List all available datasets in the base path
        """
        print("Available datasets:")
        
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir():
                    # Count images in the dataset
                    image_count = 0
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_count += 1
                    
                    print(f"  - {item.name}: {image_count} images")
        else:
            print("  No datasets found. Run download functions first.")

def main():
    """
    Main function to download datasets
    """
    downloader = DatasetDownloader()
    
    # Download all datasets
    downloader.download_all_datasets()
    
    # List available datasets
    downloader.list_available_datasets()

if __name__ == "__main__":
    main()
