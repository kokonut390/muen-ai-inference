# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install PyTorch (CPU version) first
# Using the CPU-only build significantly reduces the image size compared to the CUDA version.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install other dependencies
# Copy requirements first to leverage Docker cache layers.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application source code
COPY . .

# Expose port 5000 for the API
EXPOSE 5000

# 4. Start the FastAPI application
# Uses python -m uvicorn to ensure the module is run within the correct environment context.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]