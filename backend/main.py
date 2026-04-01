"""
FastAPI backend for crop disease detection.
Serves predictions from the trained MobileNetV2 model.
"""

import os
import json
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(
    title="Crop Disease Detection API",
    description="Detect diseases in crop leaves using deep learning",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and labels
model = None
class_labels = None
IMG_SIZE = 224

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_disease_model.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "class_labels.json")


def load_model():
    """Load the trained model and class labels."""
    global model, class_labels

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please run train_model.py first.")

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print(f"Loaded {len(class_labels)} class labels")
    else:
        print(f"WARNING: Labels file not found at {LABELS_PATH}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess an image for model prediction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "labels_loaded": class_labels is not None
    }


@app.get("/crops")
async def get_crops():
    """Return supported crops and their diseases."""
    crops = {
        "Sugarcane": {
            "diseases": ["Red Rot", "Smut", "Rust"],
            "description": "Common sugarcane leaf diseases",
            "icon": "🌾"
        },
        "Pulses": {
            "diseases": ["Anthracnose", "Powdery Mildew", "Rust"],
            "description": "Common pulse crop diseases",
            "icon": "🫘"
        },
        "Maize": {
            "diseases": ["Northern Leaf Blight", "Common Rust", "Gray Leaf Spot"],
            "description": "Common maize/corn diseases",
            "icon": "🌽"
        },
        "Wheat": {
            "diseases": ["Leaf Rust", "Septoria", "Yellow Rust"],
            "description": "Common wheat diseases",
            "icon": "🌾"
        },
        "Paddy": {
            "diseases": ["Blast", "Brown Spot", "Leaf Scald"],
            "description": "Common paddy/rice diseases",
            "icon": "🌾"
        },
        "Mustard": {
            "diseases": ["White Rust", "Alternaria Blight", "Downy Mildew"],
            "description": "Common mustard diseases",
            "icon": "🌿"
        }
    }
    return crops


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict crop disease from an uploaded image."""
    if model is None or class_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running train_model.py"
        )

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get class info
        class_info = class_labels.get(str(predicted_class_idx), {})
        crop = class_info.get("crop", "Unknown")
        disease = class_info.get("disease", "Unknown")
        class_name = class_info.get("class_name", "Unknown")

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            info = class_labels.get(str(idx), {})
            top_3.append({
                "crop": info.get("crop", "Unknown"),
                "disease": info.get("disease", "Unknown"),
                "confidence": float(predictions[0][idx])
            })

        return JSONResponse(content={
            "success": True,
            "prediction": {
                "crop": crop,
                "disease": disease,
                "class_name": class_name,
                "confidence": round(confidence * 100, 2),
                "is_healthy": disease.lower() == "healthy"
            },
            "top_predictions": top_3
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
