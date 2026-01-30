"""
Muen AI Inference Service
-------------------------
Main application entry point using FastAPI.
This service hosts a hybrid CNN-Transformer model for handwritten digit recognition.
It provides a RESTful API endpoint '/predict' for real-time inference.

Author: Kevin
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
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
        
        # CNN Block: Extracts spatial features from the image
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Linear Projection: Maps CNN output to Transformer input dimension
        # Input dim based on feature map size: 14x14 flattened
        self.linear_in = nn.Linear(14 * 14, 128)
        
        # Transformer Block: Processes sequence data
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # Classification Head: Final output layer for 10 digits
        self.fc = nn.Linear(64 * 128, 10)

    def forward(self, x):
        # 1. Feature Extraction (CNN)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 2. Reshape for Transformer
        # Treat each channel as a token in the sequence
        # Current shape: [batch, 64, 14, 14] -> [batch, 64, 196]
        x = x.view(x.size(0), 64, -1)
        
        # Project to d_model dimension: [batch, 64, 128]
        x = self.linear_in(x)
        
        # Permute for Transformer expectation: [Seq_Len, Batch, Dim]
        x = x.permute(1, 0, 2)
        
        # 3. Sequence Modeling (Transformer)
        x = self.transformer(x)
        
        # Permute back: [Batch, Seq_Len, Dim]
        x = x.permute(1, 0, 2)
        
        # Flatten for Fully Connected layer: [Batch, 64*128]
        x = x.reshape(x.size(0), -1)
        
        # 4. Classification
        x = self.fc(x)
        return x

# --- 2. Initialize FastAPI Application ---
app = FastAPI(title="Muen AI Inference API", description="API for predicting handwritten digits.")

# Global variables for model storage
model = None
device = torch.device('cpu')  # CPU is sufficient for this inference task

# --- 3. Startup Event Handler ---
@app.on_event("startup")
def load_model():
    """
    Loads the pre-trained model weights upon application startup.
    This prevents reloading the model for every request, ensuring low latency.
    """
    global model
    try:
        model = CNNTransformer()
        # Load weights and map to CPU to avoid CUDA errors in container
        # Note: weights_only=True is recommended for security in newer PyTorch versions, 
        # but we keep default behavior for compatibility.
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
        model.to(device)
        model.eval()  # Set model to evaluation mode
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# --- 4. Preprocessing Pipeline ---
# Resize to 28x28 to match training input, convert to Tensor
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# --- 5. Prediction Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to process an image file and return the predicted digit.
    
    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: Contains filename and predicted class index.
    """
    # 1. Read and Convert Image
    image_data = await file.read()
    # Convert to Grayscale ('L') to match model input channel (1)
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # 2. Preprocess
    # Add batch dimension: [1, 1, 28, 28]
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        
    # 4. Return Result
    return {
        "filename": file.filename,
        "prediction": prediction
    }