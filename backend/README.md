# Plant Disease Detection Backend

This backend contains the machine learning components for plant disease detection, including data preprocessing, model training, and inference.

## Project Structure

```
backend/
├── data_preprocessing/
│   └── data_loader.py          # Data loading and preprocessing
├── model_training/
│   ├── model_architecture.py   # CNN and transfer learning models
│   └── train_model.py         # Training pipeline
├── download_datasets.py        # Dataset downloader
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python download_datasets.py
```

This will download:
- PlantVillage dataset (38 plant species, 26 diseases)
- Rice Diseases dataset
- Plant Pathology 2020 dataset

### 3. Train the Model

```bash
cd model_training
python train_model.py
```

## Dataset Information

### PlantVillage Dataset
- **Size**: ~54,000 images
- **Classes**: 38 plant species with various diseases
- **Format**: Organized by plant type and disease
- **Example**: `Tomato___Bacterial_spot/`, `Tomato___healthy/`

### Rice Diseases Dataset
- **Size**: ~4,000 images
- **Focus**: Rice diseases
- **Classes**: Bacterial blight, Brown spot, Leaf smut, Healthy

### PlantDoc Dataset
- **Size**: ~2,600 images
- **Classes**: 13 plant species
- **Parts**: Leaves, fruits, stems

## Model Architectures

### 1. Custom CNN
- 5 convolutional blocks
- Batch normalization
- Global average pooling
- Dense layers with dropout

### 2. Transfer Learning
- **MobileNetV2**: Fast, mobile-friendly
- **EfficientNetB0**: Efficient and accurate
- **ResNet50**: Deep residual network

## Training Process

1. **Data Loading**: Load and preprocess images
2. **Data Augmentation**: Random flips, rotations, zoom, brightness
3. **Model Creation**: Choose architecture (custom or transfer learning)
4. **Training**: Train with callbacks (early stopping, model checkpoint)
5. **Fine-tuning**: Unfreeze top layers for better accuracy
6. **Evaluation**: Generate accuracy, confusion matrix, classification report

## Usage Examples

### Load and Preprocess Data

```python
from data_preprocessing.data_loader import PlantDiseaseDataLoader

# Initialize data loader
data_loader = PlantDiseaseDataLoader(
    data_path="datasets/plantvillage",
    img_size=(224, 224)
)

# Load PlantVillage dataset
data_loader.load_plantvillage_data()

# Create train-test split
X_train, X_test, y_train, y_test = data_loader.create_train_test_split()
```

### Create and Train Model

```python
from model_training.model_architecture import PlantDiseaseModel

# Create model
model_builder = PlantDiseaseModel(num_classes=38)
model = model_builder.create_transfer_learning_model(base_model='mobilenet')
model = model_builder.compile_model(model)

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
```

### Complete Training Pipeline

```python
from model_training.train_model import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    data_path="datasets/plantvillage",
    model_save_path="models/",
    results_path="results/"
)

# Complete training pipeline
trainer.load_and_prepare_data(dataset_type="plantvillage")
trainer.create_model(model_type="transfer_learning", base_model="mobilenet")
trainer.train_model(epochs=50, batch_size=32, use_augmentation=True)
trainer.fine_tune_model(epochs=20)
trainer.evaluate_model()
```

## Configuration

Modify the configuration in `train_model.py`:

```python
config = {
    "data_path": "datasets/plantvillage",
    "dataset_type": "plantvillage",
    "model_type": "transfer_learning",
    "base_model": "mobilenet",
    "epochs": 50,
    "batch_size": 32,
    "use_augmentation": True,
    "fine_tune": True,
    "fine_tune_epochs": 20
}
```

## Output Files

After training, you'll get:

- `models/best_model.h5`: Best model weights
- `models/fine_tuned_model.h5`: Fine-tuned model
- `results/class_names.json`: Class names mapping
- `results/label_encoder.pkl`: Label encoder
- `results/classification_report.json`: Detailed accuracy report
- `results/confusion_matrix.png`: Confusion matrix visualization
- `results/training_history.png`: Training curves
- `results/model_info.json`: Model information

## Next Steps

1. Train the model with your dataset
2. Evaluate performance and adjust hyperparameters
3. Save the trained model for inference
4. Integrate with FastAPI backend for real-time predictions
5. Connect with React frontend for user interface

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Use GPU or reduce model complexity
3. **Poor Accuracy**: Try different architectures or more data augmentation
4. **Dataset Issues**: Check file paths and image formats

### Performance Tips

1. Use GPU for faster training
2. Enable mixed precision training
3. Use data generators for large datasets
4. Monitor training with TensorBoard
