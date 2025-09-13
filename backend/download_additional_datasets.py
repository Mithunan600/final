import os
import kagglehub
import requests
from pathlib import Path
import zipfile
import shutil

class AdditionalDatasetDownloader:
    def __init__(self, base_path="datasets/"):
        """
        Initialize dataset downloader for additional plant disease datasets
        
        Args:
            base_path (str): Base path to store downloaded datasets
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def download_rice_diseases_dataset(self):
        """
        Download Rice Diseases dataset from Kaggle
        """
        print("🌾 Downloading Rice Diseases dataset...")
        
        try:
            path = kagglehub.dataset_download("minhhuy2810/rice-diseases-image-dataset")
            
            target_path = self.base_path / "rice_diseases"
            if not target_path.exists():
                target_path.mkdir(parents=True)
                
                # Copy files to organized structure
                if os.path.exists(path):
                    for item in os.listdir(path):
                        if os.path.isdir(os.path.join(path, item)):
                            dataset_folder = os.path.join(path, item)
                            break
                    else:
                        dataset_folder = path
                    
                    shutil.copytree(dataset_folder, target_path, dirs_exist_ok=True)
                    print(f"✅ Rice Diseases dataset saved to: {target_path}")
                else:
                    print(f"📁 Downloaded dataset found at: {path}")
            else:
                print("⚠️  Rice Diseases dataset already exists!")
                
        except Exception as e:
            print(f"❌ Error downloading Rice Diseases dataset: {e}")
    
    def download_new_plant_diseases_dataset(self):
        """
        Download New Plant Diseases dataset from Kaggle
        """
        print("🌿 Downloading New Plant Diseases dataset...")
        
        try:
            path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
            
            target_path = self.base_path / "new_plant_diseases"
            if not target_path.exists():
                target_path.mkdir(parents=True)
                
                if os.path.exists(path):
                    for item in os.listdir(path):
                        if os.path.isdir(os.path.join(path, item)):
                            dataset_folder = os.path.join(path, item)
                            break
                    else:
                        dataset_folder = path
                    
                    shutil.copytree(dataset_folder, target_path, dirs_exist_ok=True)
                    print(f"✅ New Plant Diseases dataset saved to: {target_path}")
                else:
                    print(f"📁 Downloaded dataset found at: {path}")
            else:
                print("⚠️  New Plant Diseases dataset already exists!")
                
        except Exception as e:
            print(f"❌ Error downloading New Plant Diseases dataset: {e}")
    
    def download_plant_pathology_2020(self):
        """
        Download Plant Pathology 2020 dataset from Kaggle
        """
        print("🍎 Downloading Plant Pathology 2020 dataset...")
        
        try:
            path = kagglehub.dataset_download("c/plant-pathology-2020-fgvc7")
            
            target_path = self.base_path / "plant_pathology_2020"
            if not target_path.exists():
                target_path.mkdir(parents=True)
                
                if os.path.exists(path):
                    for item in os.listdir(path):
                        if os.path.isdir(os.path.join(path, item)):
                            dataset_folder = os.path.join(path, item)
                            break
                    else:
                        dataset_folder = path
                    
                    shutil.copytree(dataset_folder, target_path, dirs_exist_ok=True)
                    print(f"✅ Plant Pathology 2020 dataset saved to: {target_path}")
                else:
                    print(f"📁 Downloaded dataset found at: {path}")
            else:
                print("⚠️  Plant Pathology 2020 dataset already exists!")
                
        except Exception as e:
            print(f"❌ Error downloading Plant Pathology 2020 dataset: {e}")
    
    def download_plantdoc_dataset(self):
        """
        Download PlantDoc dataset (if available)
        """
        print("📚 Downloading PlantDoc dataset...")
        
        try:
            # PlantDoc might be available through different sources
            # This is a placeholder - you might need to download manually
            target_path = self.base_path / "plantdoc"
            target_path.mkdir(exist_ok=True)
            
            print("📋 PlantDoc dataset information:")
            print("   - Available at: https://github.com/pratikkayal/PlantDoc-Dataset")
            print("   - Contains: 13 plant species, leaves/fruits/stems")
            print("   - Size: ~2,600 images")
            print("   - Please download manually from the GitHub repository")
            
        except Exception as e:
            print(f"❌ Error with PlantDoc dataset: {e}")
    
    def download_all_additional_datasets(self):
        """
        Download all available additional datasets
        """
        print("🌱 Downloading additional plant disease datasets...")
        print("=" * 60)
        
        self.download_rice_diseases_dataset()
        print()
        
        self.download_new_plant_diseases_dataset()
        print()
        
        self.download_plant_pathology_2020()
        print()
        
        self.download_plantdoc_dataset()
        print()
        
        print("=" * 60)
        print("✅ Additional dataset download process completed!")
        
    def list_all_datasets(self):
        """
        List all available datasets in the base path
        """
        print("📊 Available datasets:")
        print("=" * 40)
        
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir():
                    # Count images in the dataset
                    image_count = 0
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_count += 1
                    
                    print(f"  📁 {item.name}: {image_count} images")
        else:
            print("  ❌ No datasets found. Run download functions first.")
        
        print("=" * 40)

def main():
    """
    Main function to download additional datasets
    """
    downloader = AdditionalDatasetDownloader()
    
    # Download all additional datasets
    downloader.download_all_additional_datasets()
    
    # List available datasets
    downloader.list_all_datasets()

if __name__ == "__main__":
    main()
