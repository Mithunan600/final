import os
import requests
from pathlib import Path
import json
from urllib.parse import urljoin
import time

class DataCollector:
    def __init__(self, save_path="datasets/collected_data/"):
        """
        Initialize data collector for gathering more plant disease images
        
        Args:
            save_path (str): Path to save collected images
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def create_data_collection_guide(self):
        """
        Create a guide for collecting more plant disease data
        """
        guide_content = """
# 🌱 Plant Disease Data Collection Guide

## 📸 Where to Find More Plant Disease Images

### 1. **Online Sources**
- **Plant Pathology Journals**: Research papers with high-quality images
- **Agricultural Extension Services**: University and government resources
- **Plant Disease Databases**: USDA, FAO, and international databases
- **Social Media**: #PlantDisease, #PlantPathology hashtags
- **Plant Identification Apps**: iNaturalist, PlantNet, Seek

### 2. **Academic Resources**
- **Google Scholar**: Search for plant disease research papers
- **ResearchGate**: Academic papers with datasets
- **University Websites**: Agricultural extension services
- **Botanical Gardens**: Plant disease collections

### 3. **Government/Institutional Sources**
- **USDA Plant Disease Database**
- **FAO Plant Protection**
- **National Plant Diagnostic Network**
- **Extension Services**: State agricultural universities

### 4. **Commercial Sources**
- **Agricultural Companies**: Bayer, Syngenta, Corteva
- **Seed Companies**: Disease-resistant variety documentation
- **Crop Protection Companies**: Product documentation

## 📋 Data Collection Best Practices

### Image Quality Requirements:
- **Resolution**: Minimum 224x224 pixels (preferably higher)
- **Lighting**: Well-lit, natural lighting preferred
- **Focus**: Clear, sharp images of affected plant parts
- **Background**: Clean, uncluttered backgrounds
- **Angle**: Multiple angles of the same disease

### Image Categories to Collect:
1. **Early Stage Symptoms**: Initial signs of disease
2. **Advanced Symptoms**: Fully developed disease symptoms
3. **Different Plant Parts**: Leaves, stems, fruits, flowers
4. **Various Lighting**: Different times of day, weather conditions
5. **Different Varieties**: Same disease on different plant varieties

### File Organization:
```
collected_data/
├── Apple/
│   ├── Apple_Scab/
│   ├── Apple_Black_Rot/
│   ├── Apple_Cedar_Apple_Rust/
│   └── Apple_Healthy/
├── Corn/
│   ├── Corn_Common_Rust/
│   ├── Corn_Northern_Leaf_Blight/
│   └── Corn_Healthy/
└── ...
```

## 🔍 Specific Plant Diseases to Focus On

### High Priority (Common but Underrepresented):
1. **Early Blight** (Tomato, Potato)
2. **Late Blight** (Tomato, Potato)
3. **Powdery Mildew** (Various crops)
4. **Downy Mildew** (Various crops)
5. **Bacterial Spot** (Tomato, Pepper)

### Medium Priority:
1. **Anthracnose** (Various crops)
2. **Fusarium Wilt** (Tomato, Banana)
3. **Verticillium Wilt** (Various crops)
4. **Root Rot** (Various crops)

### Low Priority (Well Represented):
1. **Rust Diseases** (Already good coverage)
2. **Common Plant Diseases** (Well documented)

## 📱 Mobile Data Collection

### Smartphone Apps for Collection:
1. **iNaturalist**: Upload with GPS and identification
2. **PlantNet**: Plant identification with disease symptoms
3. **Seek**: AI-powered plant and disease identification
4. **Agrio**: Professional plant disease identification

### Collection Tips:
- Take multiple photos of the same plant
- Include close-up and wide-angle shots
- Note location, date, and plant variety
- Include healthy plants for comparison

## 🌐 Online Data Sources

### Free Datasets:
1. **PlantVillage Dataset** (Already used)
2. **PlantDoc Dataset**: More diverse plant parts
3. **iNaturalist Dataset**: Real-world plant images
4. **PlantNet Dataset**: Large-scale plant identification

### Research Papers with Datasets:
1. Search Google Scholar for "plant disease dataset"
2. Look for papers with "supplementary data"
3. Check research repositories (Zenodo, Figshare)

## 🛠️ Tools for Data Collection

### Web Scraping (Legal Sources Only):
- **BeautifulSoup**: For parsing web pages
- **Selenium**: For dynamic content
- **Scrapy**: For large-scale scraping

### Image Processing:
- **OpenCV**: Image preprocessing
- **PIL/Pillow**: Image manipulation
- **ImageMagick**: Command-line image processing

## ⚖️ Legal and Ethical Considerations

### Copyright:
- Only collect images from public domain sources
- Respect copyright and attribution requirements
- Use Creative Commons licensed images
- Get permission for commercial use

### Data Privacy:
- Don't collect images from private property without permission
- Respect farmers' privacy and property rights
- Follow local laws and regulations

## 📊 Data Augmentation Techniques

### Synthetic Data Generation:
1. **Image Rotation**: Rotate images at various angles
2. **Color Jittering**: Adjust brightness, contrast, saturation
3. **Noise Addition**: Add realistic noise patterns
4. **Geometric Transformations**: Flip, crop, resize
5. **Weather Effects**: Add rain, fog, shadows

### Advanced Techniques:
1. **GANs**: Generate synthetic plant disease images
2. **Style Transfer**: Apply different artistic styles
3. **Mixup**: Blend images from different classes
4. **CutMix**: Cut and paste regions between images

## 🔄 Continuous Improvement

### Regular Updates:
- Collect new images monthly
- Update model with new data quarterly
- Monitor model performance on new images
- Retrain model when accuracy drops

### Feedback Loop:
- Track prediction accuracy on new images
- Identify classes with low accuracy
- Focus data collection on underrepresented classes
- Validate predictions with plant pathology experts

## 📞 Expert Consultation

### Plant Pathologists:
- Contact local agricultural universities
- Reach out to extension services
- Consult with plant disease specialists
- Join plant pathology forums and groups

### Agricultural Communities:
- Join farming communities online
- Participate in agricultural forums
- Connect with crop consultants
- Engage with agricultural companies

Remember: Quality over quantity! Better to have fewer, high-quality, diverse images than many similar ones.
        """
        
        with open(self.save_path / "data_collection_guide.md", "w", encoding="utf-8") as f:
            f.write(guide_content)
        
        print("📚 Data collection guide created!")
        print(f"   Location: {self.save_path / 'data_collection_guide.md'}")
    
    def create_dataset_structure(self):
        """
        Create organized folder structure for new data
        """
        # Create main categories
        categories = [
            "Apple", "Blueberry", "Cherry", "Corn", "Grape",
            "Tomato", "Potato", "Pepper", "Cucumber", "Lettuce",
            "Strawberry", "Raspberry", "Peach", "Pear", "Plum"
        ]
        
        diseases = [
            "Healthy", "Scab", "Black_Rot", "Rust", "Mildew",
            "Blight", "Spot", "Wilt", "Rot", "Anthracnose",
            "Powdery_Mildew", "Downy_Mildew", "Bacterial_Spot",
            "Fusarium_Wilt", "Verticillium_Wilt"
        ]
        
        for category in categories:
            category_path = self.save_path / category
            category_path.mkdir(exist_ok=True)
            
            for disease in diseases:
                disease_path = category_path / f"{category}_{disease}"
                disease_path.mkdir(exist_ok=True)
                
                # Create a README file for each disease folder
                readme_content = f"""# {category} - {disease.replace('_', ' ')}

## Image Requirements:
- Minimum resolution: 224x224 pixels
- Clear, well-lit images
- Multiple angles if possible
- Include both close-up and wide shots

## File Naming:
- Use descriptive names: {category.lower()}_{disease.lower()}_001.jpg
- Include date if relevant: {category.lower()}_{disease.lower()}_2024_001.jpg

## Notes:
- Add any relevant information about the image
- Include location if known
- Note plant variety if available
"""
                
                with open(disease_path / "README.md", "w") as f:
                    f.write(readme_content)
        
        print("📁 Dataset structure created!")
        print(f"   Location: {self.save_path}")
        print(f"   Categories: {len(categories)}")
        print(f"   Diseases per category: {len(diseases)}")
    
    def create_collection_checklist(self):
        """
        Create a checklist for data collection
        """
        checklist_content = """
# ✅ Plant Disease Data Collection Checklist

## Before Starting:
- [ ] Review data collection guide
- [ ] Set up folder structure
- [ ] Install necessary tools
- [ ] Review legal requirements
- [ ] Plan collection strategy

## During Collection:
- [ ] Take multiple angles of each plant
- [ ] Include healthy plants for comparison
- [ ] Note location and date
- [ ] Record plant variety if known
- [ ] Ensure good lighting
- [ ] Check image quality before saving

## Image Quality Checks:
- [ ] Image is in focus
- [ ] Disease symptoms are clearly visible
- [ ] Background is not cluttered
- [ ] Image resolution is adequate (224x224+)
- [ ] Color representation is accurate

## File Organization:
- [ ] Images saved in correct folders
- [ ] Descriptive file names used
- [ ] README files updated
- [ ] Metadata recorded if available

## After Collection:
- [ ] Review collected images
- [ ] Remove low-quality images
- [ ] Organize by disease severity
- [ ] Create backup copies
- [ ] Update dataset documentation

## Weekly Tasks:
- [ ] Review collection progress
- [ ] Identify gaps in coverage
- [ ] Plan next collection targets
- [ ] Update collection strategy

## Monthly Tasks:
- [ ] Analyze dataset balance
- [ ] Retrain model if needed
- [ ] Evaluate model performance
- [ ] Plan data augmentation

## Quality Metrics:
- [ ] Minimum 100 images per class
- [ ] Balanced representation across classes
- [ ] Diverse lighting conditions
- [ ] Multiple plant varieties per disease
- [ ] Various disease severities
        """
        
        with open(self.save_path / "collection_checklist.md", "w", encoding="utf-8") as f:
            f.write(checklist_content)
        
        print("📋 Collection checklist created!")
        print(f"   Location: {self.save_path / 'collection_checklist.md'}")

def main():
    """
    Main function to set up data collection
    """
    collector = DataCollector()
    
    print("🌱 Setting up plant disease data collection...")
    print("=" * 50)
    
    # Create data collection guide
    collector.create_data_collection_guide()
    
    # Create dataset structure
    collector.create_dataset_structure()
    
    # Create collection checklist
    collector.create_collection_checklist()
    
    print("\n✅ Data collection setup completed!")
    print("\n📖 Next steps:")
    print("1. Read the data collection guide")
    print("2. Review the collection checklist")
    print("3. Start collecting images systematically")
    print("4. Use the organized folder structure")
    print("5. Focus on underrepresented disease classes")

if __name__ == "__main__":
    main()
