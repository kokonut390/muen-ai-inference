from fastapi.testclient import TestClient
from main import app, load_model # <--- 1. 多 import load_model
from PIL import Image
import io
import pytest

# 初始化測試客戶端
client = TestClient(app)

# --- 關鍵修正：在測試開始前，強制載入模型 ---
# 因為 TestClient 不會自動觸發 @app.on_event("startup")，所以我們要手動呼叫
load_model()
# ----------------------------------------

def create_dummy_image():
    """建立一張 28x28 的全黑測試圖片"""
    img = Image.new('L', (28, 28), color=0)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_api_documentation():
    """測試 Swagger UI 是否正常開啟"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_predict_single():
    """測試單張圖片推論 API"""
    img_bytes = create_dummy_image()
    files = {'file': ('test.png', img_bytes, 'image/png')}
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "filename" in json_data
    assert isinstance(json_data["prediction"], int)

def test_predict_batch():
    """測試批次推論 API (Bonus Feature)"""
    img1 = create_dummy_image()
    img2 = create_dummy_image()
    
    files = [
        ('files', ('batch1.png', img1, 'image/png')),
        ('files', ('batch2.png', img2, 'image/png'))
    ]
    
    response = client.post("/predict_batch", files=files)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "results" in json_data
    assert len(json_data["results"]) == 2
    assert json_data["results"][0]["status"] == "success"