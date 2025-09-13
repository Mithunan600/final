# 🌱 Plant Disease Detection System

An AI-powered plant disease detection system using deep learning, built with React frontend and FastAPI backend.

## 🎯 Features

- **98.2% Accuracy**: Highly accurate disease detection across 15 plant diseases
- **Real-time Analysis**: Instant image processing and disease prediction
- **Modern UI**: Beautiful, responsive React frontend with drag-and-drop
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Memory Efficient**: Optimized for large datasets without memory issues
- **Production Ready**: Complete system ready for deployment

## 🏗️ Architecture

```
Plant Disease Detection System
├── Frontend (React)
│   ├── Modern UI with drag-and-drop
│   ├── Real-time image analysis
│   └── Detailed results display
├── Backend (FastAPI)
│   ├── RESTful API endpoints
│   ├── Image preprocessing
│   └── Model inference
└── ML Model (TensorFlow/Keras)
    ├── MobileNetV2 transfer learning
    ├── 15 disease classes
    └── 98.2% accuracy
```

## 📊 Model Performance

### Overall Accuracy: **98.20%**

### Supported Plant Diseases:
- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Corn/Maize**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape**: Black Rot, Esca (Black Measles), Leaf Blight, Healthy

### Per-Class Performance:
- **Perfect (100%)**: Corn Common Rust, Corn Healthy, Grape Leaf Blight
- **Excellent (95-99%)**: Most Apple, Cherry, and Grape diseases
- **Good (85%+)**: All classes above 84% accuracy

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or download the project
cd finalYearPrj

# Run the startup script (installs dependencies and starts both servers)
python start_app.py
```

This will:
1. Install all dependencies
2. Start the FastAPI backend (port 8000)
3. Start the React frontend (port 3000)
4. Open the app in your browser

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
cd api
python app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📁 Project Structure

```
finalYearPrj/
├── backend/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   ├── test_api.py         # API testing script
│   │   └── requirements.txt    # Backend dependencies
│   ├── data_preprocessing/
│   │   ├── data_loader.py      # Dataset loading utilities
│   │   └── data_generator.py   # Memory-efficient data generators
│   ├── model_training/
│   │   ├── train_model.py      # Training pipeline
│   │   ├── model_architecture.py # CNN and transfer learning models
│   │   ├── evaluate_model.py   # Model evaluation script
│   │   └── check_training_status.py # Training status checker
│   ├── datasets/
│   │   └── plantvillage/
│   │       └── color/          # PlantVillage dataset (14,021 images)
│   ├── models/
│   │   └── best_model.keras    # Trained model (98.2% accuracy)
│   └── results/                # Training results and evaluation reports
├── frontend/
│   ├── src/
│   │   ├── App.js              # Main React application
│   │   ├── index.js            # React entry point
│   │   └── index.css           # Global styles
│   ├── public/
│   │   ├── index.html          # HTML template
│   │   └── manifest.json       # PWA manifest
│   └── package.json            # Frontend dependencies
├── start_app.py                # Automated startup script
└── README.md                   # This file
```

## 🔧 API Endpoints

### Health & Info
- `GET /` - API information
- `GET /health` - Health check
- `GET /classes` - List supported disease classes

### Prediction
- `POST /predict` - Analyze single image
- `POST /predict-batch` - Analyze multiple images (max 10)

### Example API Usage
```bash
# Health check
curl http://localhost:8000/health

# Get supported classes
curl http://localhost:8000/classes

# Predict disease from image
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@plant_image.jpg"
```

## 🎨 Frontend Features

- **Drag & Drop Upload**: Easy image upload interface
- **Real-time Preview**: Instant image preview
- **Confidence Visualization**: Confidence bars and percentages
- **Top 3 Predictions**: Shows multiple possible diagnoses
- **Responsive Design**: Works on all devices
- **Error Handling**: User-friendly error messages

## 🔬 Technical Details

### Model Architecture
- **Base Model**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224x3 RGB images
- **Parameters**: 2.4M total (165K trainable)
- **Framework**: TensorFlow 2.17 / Keras
- **Optimization**: Adam optimizer with learning rate scheduling

### Data Processing
- **Dataset**: PlantVillage (14,021 images, 15 classes)
- **Preprocessing**: Resize, normalize, augment
- **Memory Efficient**: Generators for large datasets
- **Train/Test Split**: 80/20 stratified split

### Performance Optimizations
- **Transfer Learning**: Pre-trained MobileNetV2 weights
- **Data Augmentation**: Rotation, flip, zoom, brightness
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model during training

## 📱 Usage Instructions

1. **Start the Application**:
   ```bash
   python start_app.py
   ```

2. **Upload Plant Image**:
   - Drag and drop an image onto the upload area
   - Or click to select a file from your device

3. **Analyze Disease**:
   - Click "Analyze Plant Disease" button
   - Wait for AI processing (usually 2-5 seconds)

4. **View Results**:
   - See predicted disease with confidence score
   - View top 3 predictions
   - Check if plant is healthy or diseased

## 🛠️ Development

### Training New Models
```bash
cd backend/model_training
python train_model.py
```

### Evaluating Models
```bash
cd backend/model_training
python evaluate_model.py
```

### Testing API
```bash
cd backend/api
python test_api.py
```

## 📊 Dataset Information

- **Source**: PlantVillage Dataset
- **Images**: 14,021 total
- **Classes**: 15 disease types
- **Plants**: Apple, Blueberry, Cherry, Corn, Grape
- **Format**: JPG images, organized by plant type and disease

## 🚀 Deployment

### Backend Deployment
- Deploy FastAPI app to cloud platforms (AWS, GCP, Azure)
- Use Docker for containerization
- Set up proper CORS for production

### Frontend Deployment
- Build production version: `npm run build`
- Deploy to static hosting (Netlify, Vercel, AWS S3)
- Configure proxy for API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of a final year academic project for plant disease detection using AI/ML.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing comprehensive plant disease images
- **TensorFlow/Keras**: For deep learning framework
- **FastAPI**: For modern Python web framework
- **React**: For frontend framework
- **Transfer Learning**: MobileNetV2 pre-trained weights

## 📞 Support

For questions or issues:
1. Check the troubleshooting section in individual README files
2. Review the API documentation
3. Test with sample images from the dataset

---

**Built with ❤️ using AI/ML for better agriculture**
