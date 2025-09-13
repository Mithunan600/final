from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import json
import os
from pathlib import Path

# Load the trained model and class names
MODEL_PATH = "../model_training/models/best_model.keras"
CLASS_NAMES_PATH = "../model_training/results/class_names.json"

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease detection using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and class names
model = None
class_names = None

def load_model():
    """Load the trained model and class names"""
    global model, class_names
    
    try:
        # Load model
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Load class names
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} class names")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model prediction
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        Preprocessed image array
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Resize to model input size
        image_array = cv2.resize(image_array, (224, 224))
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_disease(image_array):
    """
    Predict plant disease from preprocessed image
    
    Args:
        image_array: Preprocessed image array
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get predicted class index and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            top_3_predictions.append({
                "class": class_names[idx],
                "confidence": float(predictions[0][idx])
            })
        
        # Parse plant type and disease
        if "_" in predicted_class:
            plant_type, disease = predicted_class.split("_", 1)
        else:
            plant_type = "Unknown"
            disease = predicted_class
        
        return {
            "predicted_class": predicted_class,
            "plant_type": plant_type,
            "disease": disease,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "is_healthy": "healthy" in predicted_class.lower()
        }
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "class_names_loaded": class_names is not None
    }

@app.get("/classes")
async def get_classes():
    """Get list of supported plant disease classes"""
    if not class_names:
        raise HTTPException(status_code=500, detail="Class names not loaded")
    
    return {
        "classes": class_names,
        "count": len(class_names)
    }

@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_disease(processed_image)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "model_info": {
                "model_path": MODEL_PATH,
                "num_classes": len(class_names)
            }
        })
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict plant diseases from multiple uploaded images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with batch prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Read and process image
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            result = predict_disease(processed_image)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_files": len(files),
        "successful_predictions": sum(1 for r in results if r["success"]),
        "results": results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
