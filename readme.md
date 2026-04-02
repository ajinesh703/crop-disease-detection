# 🌱 CropGuard AI — Crop Disease Detection

An AI-powered web application that detects crop leaf diseases using deep learning. Upload a photo of a crop leaf and get instant disease diagnosis with confidence scores.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)

---

## ✨ Features

- **96%+ Validation Accuracy** — MobileNetV2 transfer learning trained on PlantVillage dataset
- **38 Disease Classes** — Covers multiple crops and their common diseases
- **Real-time Inference** — FastAPI backend serves predictions in under 1 second
- **Drag & Drop UI** — Clean React + TypeScript frontend with confidence visualization
- **Top-3 Predictions** — Shows alternative predictions with confidence scores
- **Low Confidence Guard** — Rejects unclear images with helpful feedback

---

## 🗂️ Project Structure

```
crop-disease-detection/
├── backend/
│   ├── train_model.py          # MobileNetV2 training script (2-phase)
│   ├── main.py                 # FastAPI server
│   ├── crop_disease_model.h5   # Trained model (generated after training)
│   ├── class_labels.json       # Class index → crop/disease mapping
│   ├── requirements.txt        # Python dependencies
│   └── PlantVillage-Dataset/   # Dataset (cloned separately)
└── frontend/
    ├── src/
    │   ├── App.tsx             # Main React component
    │   ├── App.css             # Styles
    │   └── index.css           # Global styles + CSS variables
    ├── package.json
    └── vite.config.ts
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Node.js 18+
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/ajinesh703/crop-disease-detection.git
cd crop-disease-detection
```

---

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

#### Clone the PlantVillage Dataset

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
```

Make sure the folder structure looks like:
```
backend/PlantVillage-Dataset/raw/color/
    ├── Apple___Apple_scab/
    ├── Apple___healthy/
    ├── Tomato___Late_blight/
    └── ... (38 classes total)
```

#### Train the Model

```bash
python train_model.py
```

Training runs in **2 phases**:
- **Phase 1** (Epochs 1–10): Trains only the classification head, MobileNetV2 frozen
- **Phase 2** (Epochs 11–20): Fine-tunes top 30 layers of MobileNetV2

> Training takes ~3–5 hours on CPU. On GPU it's significantly faster.  
> Best model is auto-saved to `crop_disease_model.h5` (based on `val_accuracy`).

#### Start the Backend Server

```bash
python main.py
```

Server starts at: `http://localhost:8000`

API docs available at: `http://localhost:8000/docs`

---

### 3. Frontend Setup

Open a **new terminal**:

```bash
cd frontend
npm install
npm run dev
```

Frontend starts at: `http://localhost:5173`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server + model status |
| GET | `/crops` | All supported crops and diseases |
| POST | `/predict` | Upload image and get prediction |

### Example `/predict` Response

```json
{
  "success": true,
  "prediction": {
    "crop": "Tomato",
    "disease": "Late_blight",
    "class_name": "Tomato___Late_blight",
    "confidence": 94.72,
    "is_healthy": false
  },
  "top_predictions": [
    { "crop": "Tomato", "disease": "Late_blight", "confidence": 94.72 },
    { "crop": "Tomato", "disease": "Early_blight", "confidence": 3.21 },
    { "crop": "Tomato", "disease": "healthy", "confidence": 1.12 }
  ]
}
```

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| Base Model | MobileNetV2 (ImageNet weights) |
| Input Size | 224 × 224 × 3 |
| Head | GlobalAvgPool → BN → Dense(256) → Dropout(0.5) → Dense(128) → Softmax |
| Optimizer | Adam (lr=0.001 → 0.0001) |
| Loss | Categorical Crossentropy |
| Dataset | PlantVillage (color, ~54,000 images) |
| Classes | 38 |
| Val Accuracy | **96.14%** |

---

## 📦 Requirements

### Backend (`requirements.txt`)

```
fastapi
uvicorn
tensorflow
pillow
numpy
scikit-learn
```

### Frontend

```
react 18
typescript 5
vite 5
```

---

## ⚠️ Troubleshooting

**Black screen on frontend?**
Make sure `npm install` completed without errors, then restart `npm run dev`.

**Model not found error?**
Run `python train_model.py` first to generate `crop_disease_model.h5`.

**TensorFlow warnings on startup?**
These are harmless deprecation warnings — they don't affect functionality.

**Low confidence result?**
Use a clear, well-lit photo of a single leaf against a plain background.

---

## 🙌 Credits

- Dataset: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) by Mohanty et al.
- Model: MobileNetV2 via TensorFlow/Keras
- Built by [@ajinesh703](https://github.com/ajinesh703)