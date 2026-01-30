"""
Batch Prediction Script
-----------------------
This script handles the end-to-end inference process in two phases:
1. Data Preparation (ETL): Generates images from 'test.csv' if the image folder is missing.
2. Batch Inference: Iterates through the image folder and sends requests to the Inference API.

Usage:
    python batch_predict.py
"""

import pandas as pd
import numpy as np
import requests
import os
from PIL import Image
import io

# --- Configuration ---
API_URL = "http://localhost:5000/predict"    # API endpoint
CSV_PATH = "test.csv"                        # Source pixel data (provided by exam)
IMAGE_DIR = "test_images"                    # Target folder for generated images
OUTPUT_CSV = "result.csv"                    # Final output file

def generate_images_from_csv():
    """
    Phase 1: ETL (Extract, Transform, Load)
    Reads pixel data from CSV and converts it into PNG images in a directory.
    This ensures we have a valid folder of images to process, satisfying the exam requirement.
    """
    if not os.path.exists(CSV_PATH):
        print(f"⚠️ No {CSV_PATH} found. Skipping generation.")
        return

    print(f"📂 Generating images from {CSV_PATH} to {IMAGE_DIR}...")
    df = pd.read_csv(CSV_PATH)
    
    # --- Optimization for Demo ---
    # Only process the first 50 records to save time during testing.
    # [NOTE] Comment out the line below to process the full dataset for final submission.
    df = df.head(50) 
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    for idx, row in df.iterrows():
        # Reshape flat 784 pixels to 28x28 array
        pixels = row.values.astype(np.uint8).reshape(28, 28)
        img = Image.fromarray(pixels)
        
        # Naming convention: {index}.png 
        # Using index ensures we can track the order of images strictly.
        img.save(os.path.join(IMAGE_DIR, f"{idx}.png"))
    
    print(f"✅ Generated {len(df)} images.")

def predict_from_folder():
    """
    Phase 2: Inference
    Iterates through the 'test_images' directory, sends each image to the API,
    and aggregates results. This strictly follows the 'folder-based inference' requirement.
    """
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Folder {IMAGE_DIR} does not exist.")
        return

    print(f"🚀 Starting batch inference from folder: {IMAGE_DIR}...")
    
    # Filter for valid image files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # --- Critical Sorting Step ---
    # Standard os.listdir() returns arbitrary order (e.g., 1.png, 10.png, 2.png).
    # We sort numerically to ensure the output CSV corresponds to the input CSV order.
    try:
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # Fallback: Use string sorting if filenames are not numeric
        image_files.sort()

    results = []
    
    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        
        try:
            # Send HTTP POST request with the image file
            with open(img_path, "rb") as f:
                files = {"file": (img_file, f, "image/png")}
                response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                # Parse the JSON response
                pred = response.json()["prediction"]
                results.append({"ImageID": img_file, "Label": pred})
                print(f"✅ {img_file} -> {pred}")
            else:
                print(f"❌ {img_file} Failed: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Error processing {img_file}: {e}")

    # Export results to CSV
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"\n🏆 Done! Results saved to {OUTPUT_CSV}")
    else:
        print("❌ No predictions made.")

if __name__ == "__main__":
    """
    Main Execution Flow:
    1. Check if the image folder exists and is populated.
    2. If empty or missing, trigger Phase 1 (Generate from CSV).
    3. Always execute Phase 2 (Predict from Folder).
    """
    
    # Check if we need to generate images first
    if not os.path.exists(IMAGE_DIR) or not os.listdir(IMAGE_DIR):
        generate_images_from_csv()
    
    # Proceed to inference
    predict_from_folder()