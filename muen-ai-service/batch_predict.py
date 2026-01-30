import pandas as pd
import numpy as np
import requests
import os
from PIL import Image
import io

# --- Configuration ---
API_URL = "http://localhost:5000/predict"
CSV_PATH = "test.csv"
IMAGE_DIR = "test_images"  # Fixed folder name as per requirement
OUTPUT_CSV = "result.csv"

def generate_images_from_csv():
    """
    Phase 1: ETL - Generate images from CSV if folder is empty.
    """
    if not os.path.exists(CSV_PATH):
        print(f"⚠️ No {CSV_PATH} found. Skipping generation.")
        return

    print(f"📂 Generatimg images from {CSV_PATH} to {IMAGE_DIR}...")
    df = pd.read_csv(CSV_PATH)
    
    # Optimization: Process only first 50 for demo (Uncomment below to run all)
    df = df.head(50) 
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    for idx, row in df.iterrows():
        pixels = row.values.astype(np.uint8).reshape(28, 28)
        img = Image.fromarray(pixels)
        # Naming convention: {index}.png to maintain order
        img.save(os.path.join(IMAGE_DIR, f"{idx}.png"))
    
    print(f"✅ Generated {len(df)} images.")

def predict_from_folder():
    """
    Phase 2: Inference - Strictly iterate through the folder to predict.
    """
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Folder {IMAGE_DIR} does not exist.")
        return

    print(f"🚀 Starting batch inference from folder: {IMAGE_DIR}...")
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # CRITICAL: Sort files numerically (0.png, 1.png, 2.png...) 
    # Otherwise 10.png comes before 2.png in standard sorting
    try:
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # Fallback for non-numeric filenames
        image_files.sort()

    results = []
    
    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        
        try:
            with open(img_path, "rb") as f:
                files = {"file": (img_file, f, "image/png")}
                response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                pred = response.json()["prediction"]
                results.append({"ImageID": img_file, "Label": pred})
                print(f"✅ {img_file} -> {pred}")
            else:
                print(f"❌ {img_file} Failed: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Error processing {img_file}: {e}")

    # Export
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"\n🏆 Done! Results saved to {OUTPUT_CSV}")
    else:
        print("❌ No predictions made.")

if __name__ == "__main__":
    # Logic: 
    # 1. If folder doesn't exist or is empty, try to generate from CSV.
    # 2. Then, run inference on the folder.
    
    if not os.path.exists(IMAGE_DIR) or not os.listdir(IMAGE_DIR):
        generate_images_from_csv()
    
    predict_from_folder()