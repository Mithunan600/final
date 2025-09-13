import requests
import json
from pathlib import Path

def test_api():
    """Test the plant disease detection API"""
    
    base_url = "http://localhost:8000"
    
    print("Testing Plant Disease Detection API...")
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Get classes
    print("\n2. Testing get classes...")
    try:
        response = requests.get(f"{base_url}/classes")
        if response.status_code == 200:
            print("✅ Get classes passed")
            data = response.json()
            print(f"   Number of classes: {data['count']}")
            print(f"   Sample classes: {data['classes'][:3]}...")
        else:
            print(f"❌ Get classes failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Get classes error: {e}")
    
    # Test 3: Test with a sample image (if available)
    print("\n3. Testing prediction with sample image...")
    
    # Look for a sample image in the dataset
    sample_image_path = None
    dataset_path = Path("../datasets/plantvillage/color")
    
    if dataset_path.exists():
        # Find first image file
        for folder in dataset_path.iterdir():
            if folder.is_dir():
                for img_file in folder.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        sample_image_path = img_file
                        break
                if sample_image_path:
                    break
    
    if sample_image_path:
        try:
            with open(sample_image_path, 'rb') as f:
                files = {'file': (sample_image_path.name, f, 'image/jpeg')}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 200:
                print("✅ Prediction test passed")
                result = response.json()
                prediction = result['prediction']
                print(f"   Predicted class: {prediction['predicted_class']}")
                print(f"   Confidence: {prediction['confidence']:.3f}")
                print(f"   Plant type: {prediction['plant_type']}")
                print(f"   Disease: {prediction['disease']}")
                print(f"   Is healthy: {prediction['is_healthy']}")
            else:
                print(f"❌ Prediction test failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Prediction test error: {e}")
    else:
        print("⚠️  No sample image found for testing")
    
    print("\n" + "="*50)
    print("API Testing Complete!")

if __name__ == "__main__":
    test_api()
