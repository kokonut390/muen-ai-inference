import pandas as pd
import numpy as np
import requests
import os
from PIL import Image
import io

# --- Configuration ---
API_URL = "http://localhost:5000/predict"    # API endpoint inside the container network or localhost
CSV_PATH = "test.csv"                        # Source CSV file provided by the exam
IMAGE_DIR = "test_images_temp"               # Temporary directory to store generated images
OUTPUT_CSV = "result.csv"                    # Final output filename

# 1. Create temporary directory for images
# We need to simulate a folder input as per exam requirements
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

print(f"🚀 Starting batch prediction process...")

try:
    # 2. Load CSV Data
    # Assuming test.csv has the same format as train.csv (pixel0...pixel783)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Please ensure the file exists.")
        
    df = pd.read_csv(CSV_PATH)
    print(f"📄 Loaded {len(df)} records from {CSV_PATH}. Processing...")
    
    # --- Optimization: Limit to first 50 records for demonstration ---
    # Comment out the line below to process the full dataset
    df = df.head(50)
    
    results = []
    
    # 3. Iterate through each row to generate image and predict
    for idx, row in df.iterrows():
        # Convert flattened pixels (1x784) back to 28x28 image
        pixels = row.values.astype(np.uint8).reshape(28, 28)
        img = Image.fromarray(pixels)
        
        # Save as a temporary PNG file
        img_filename = f"{idx}.png"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        img.save(img_path)
        
        # 4. Send request to the Inference API
        try:
            with open(img_path, "rb") as f:
                files = {"file": (img_filename, f, "image/png")}
                response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                # Parse JSON response
                pred = response.json()["prediction"]
                
                # Append result (ImageID, Label)
                # Note: Adjust ImageID format as needed (e.g., idx+1)
                results.append({"ImageID": idx + 1, "Label": pred})
                
                # Print progress
                print(f"✅ Image {idx} -> Prediction: {pred}")
            else:
                print(f"❌ Image {idx} Failed: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Connection Error on image {idx}: {e}")

    # 5. Export results to CSV
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🏆 Batch prediction complete! Results saved to: {OUTPUT_CSV}")
    else:
        print("❌ No results were generated.")

except Exception as e:
    print(f"Critical Error: {e}")