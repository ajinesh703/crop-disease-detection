"""
FastAPI backend for crop disease detection.
Serves predictions from the trained MobileNetV2 model.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# Global model and labels
model = None
class_labels = None
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    load_model()
    yield


app = FastAPI(
    title="Crop Disease Detection API",
    description="Detect diseases in crop leaves using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "labels_loaded": class_labels is not None,
        "num_classes": len(class_labels) if class_labels else 0
    }


@app.get("/crops")
async def get_crops():
    """Return all supported crops and diseases from actual class_labels.json."""
    if class_labels is None:
        raise HTTPException(status_code=503, detail="Labels not loaded.")

    crops = {}
    for idx, info in class_labels.items():
        crop = info.get("crop", "Unknown")
        disease = info.get("disease", "Unknown")

        if crop not in crops:
            crops[crop] = {"diseases": [], "total_classes": 0}

        if disease not in crops[crop]["diseases"]:
            crops[crop]["diseases"].append(disease)

        crops[crop]["total_classes"] += 1

    return {
        "total_crops": len(crops),
        "total_classes": len(class_labels),
        "crops": crops
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict crop disease from an uploaded image."""
    if model is None or class_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run train_model.py first."
        )

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Low confidence check
        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                "success": False,
                "message": f"Low confidence ({round(confidence * 100, 2)}%). Please upload a clearer leaf image.",
                "confidence": round(confidence * 100, 2)
            })

        class_info = class_labels.get(str(predicted_class_idx), {})
        crop = class_info.get("crop", "Unknown")
        disease = class_info.get("disease", "Unknown")
        class_name = class_info.get("class_name", "Unknown")

        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            info = class_labels.get(str(idx), {})
            top_3.append({
                "crop": info.get("crop", "Unknown"),
                "disease": info.get("disease", "Unknown"),
                "confidence": round(float(predictions[0][idx]) * 100, 2)
            })

        return JSONResponse(content={
            "success": True,
            "prediction": {
                "crop": crop,
                "disease": disease,
                "class_name": class_name,
                "confidence": round(confidence * 100, 2),
                "is_healthy": "healthy" in disease.lower()
            },
            "top_predictions": top_3
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)