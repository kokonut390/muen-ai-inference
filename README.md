# Muen AI Inference Service

這是一個基於 **PyTorch** 與 **FastAPI** 的手寫數字辨識推論服務。
本專案採用 Docker 容器化技術進行部署，並提供 RESTful API 與自動化批次推論腳本，完整展示從模型部署到批次資料處理的流程。

## 📂 專案結構 (Project Structure)

```text
.
├── README.md             # 專案說明文件
├── .gitignore            # Git 忽略設定
└── muen-ai-service/      # 核心程式碼目錄
    ├── main.py           # FastAPI 服務主程式 (包含 CNN+Transformer 模型架構)
    ├── batch_predict.py  # 批次預測腳本 (ETL 流程：CSV -> Images -> Inference -> CSV)
    ├── Dockerfile        # 容器化建置設定 (基於 python:3.9-slim)
    ├── requirements.txt  # Python 相依套件清單
    └── model_weights.pth # 預先訓練好的模型權重
```

## ⚠️ 特別說明 (Important Notes)

為了符合考題要求並適應原始資料格式，本專案包含以下特別設計：

1.  **資料來源前處理 (Data Preprocessing)**：
    由於原始資料僅提供 `test.csv` (像素數據)，但題目要求「對資料夾中圖片進行推論」。
    因此，本專案的 `batch_predict.py` 腳本內建 ETL 流程，會自動讀取 CSV 並將其轉換為 `test_images/` 資料夾中的實體圖片，再進行 API 推論，完美達成題目邏輯要求。

2.  **API 測試工具 (Testing Tool)**：
    本服務整合了 **Swagger UI (OpenAPI)** 自動化文件系統。
    啟動服務後，可直接透過瀏覽器進行互動式 API 測試 (上傳圖片、檢視 JSON 回應)，其功能與效力等同於 Postman。

---

## 🚀 快速開始 (Quick Start)

### 步驟 0：準備資料 (Data Setup)
本專案不包含原始數據。在開始之前，請確保將 `test.csv` 檔案放置於專案根目錄 (`muen-ai-service/` 資料夾內)。
> ⚠️ 注意：若無此檔案，批次預測腳本將無法生成圖片進行推論。

### 步驟 1：進入專案目錄
請開啟終端機，並切換至程式碼所在的資料夾：

```bash
cd muen-ai-service
```

### 步驟 2：建置 Docker 映像檔
執行以下指令來打包應用程式 (第一次執行需下載 PyTorch，約需 1-3 分鐘)：

```bash
docker build -t muen-exam-img .
```

### 步驟 3：啟動推論服務
將容器的 5000 Port 對應到本機的 5000 Port：

```bash
docker run -p 5000:5000 muen-exam-img
```

> 當看到 `Uvicorn running on http://0.0.0.0:5000` 即代表啟動成功。

### 步驟 4：API 功能測試 (單張圖片)
請打開瀏覽器訪問： http://localhost:5000/docs ，並依照以下規範進行測試。

#### 📥 輸入格式說明 (Input Format)
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **參數 (Body)**:
  - `file`: 上傳圖片檔案 (格式支援 PNG, JPG, JPEG)

#### ✅ 輸出範例 (Output Example)
成功執行後，API 將回傳如下 JSON 格式：

```json
{
  "filename": "img_0_label_6.png",
  "prediction": 6
}
```

---

## 📦 批次預測 (Batch Prediction)

本專案提供自動化腳本，可一次性完成「圖片生成」、「API 推論」與「結果匯出」。

請保持 Docker 服務運作中，並開啟**另一個終端機視窗**執行以下指令：

#### 1. 查詢容器 ID
請複製 CONTAINER ID (例如: a1b2c3d4e5f6)
```bash
docker ps
```


#### 2. 將腳本複製到容器內
```bash
docker cp batch_predict.py <Container_ID>:/app/
```

#### 3. 執行預測腳本
此指令會自動處理圖片並呼叫 API，輸出進度條：
```bash
docker exec <Container_ID> python -u batch_predict.py
```

#### 4. 取出結果 CSV
預測完成後，將結果檔案下載回本機：
```bash
docker cp <Container_ID>:/app/result.csv .
```

現在，您可以在專案目錄下找到 `result.csv` 檔案。