"""
Muen AI Inference Service
-------------------------
Main application entry point using FastAPI.
Hosts a hybrid CNN-Transformer model for handwritten digit recognition.
Includes endpoints for single and batch image prediction.

Author: Kevin
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from typing import List
from PIL import Image
import io
import torchvision.transforms as transforms

# --- 1. Model Architecture Definition ---
class CNNTransformer(nn.Module):
    """
    A hybrid neural network architecture combining CNN for feature extraction
    and Transformer Encoder for capturing global dependencies.
    """
    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear_in = nn.Linear(14 * 14, 128)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(64 * 128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), 64, -1)
        x = self.linear_in(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# --- 2. Initialize FastAPI Application ---
app = FastAPI(
    title="Muen AI Inference API", 
    description="API for predicting handwritten digits (Single & Batch support)."
)

# Global variables
model = None
device = torch.device('cpu') 

# --- 3. Startup Event Handler ---
@app.on_event("startup")
def load_model():
    global model
    try:
        model = CNNTransformer()
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# --- 4. Preprocessing Pipeline ---
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# --- 5. Helper Function for Prediction ---
def predict_image(image_bytes):
    """Helper to process a single image byte stream."""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# --- 6. Endpoints ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Single image prediction endpoint.
    """
    image_data = await file.read()
    prediction = predict_image(image_data)
    return {
        "filename": file.filename,
        "prediction": prediction
    }

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction endpoint (Bonus Feature).
    Accepts multiple image files and returns a list of predictions.
    """
    results = []
    for file in files:
        try:
            image_data = await file.read()
            pred = predict_image(image_data)
            results.append({
                "filename": file.filename,
                "prediction": pred,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
            
    return {"results": results}