# 📖 使用指南

本文檔提供IMDB情感分析專案的詳細使用指南，包括安裝、配置、訓練和部署等步驟。

## 📋 目錄

- [環境要求](#環境要求)
- [安裝步驟](#安裝步驟)
- [資料準備](#資料準備)
- [模型訓練](#模型訓練)
- [API服務](#api服務)
- [常見問題](#常見問題)

## 🔧 環境要求

### 硬體要求
- **CPU**: 4核心以上（推薦8核心）
- **RAM**: 8GB以上（推薦16GB）
- **GPU**: NVIDIA GPU with 4GB+ VRAM（可選，用於加速訓練）
- **儲存**: 10GB可用空間

### 軟體要求
- **Python**: 3.8 - 3.11
- **操作系統**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **CUDA**: 11.8+（如使用GPU）

## 📦 安裝步驟

### 方法一：使用pip（推薦）

```bash
# 1. 克隆專案
git clone <repository-url>
cd imdb_sentiment_analysis

# 2. 建立虛擬環境
python -m venv .venv

# 3. 啟動虛擬環境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 4. 升級pip
python -m pip install --upgrade pip

# 5. 安裝依賴
pip install -r requirements.txt

# 6. 安裝spaCy語言模型
python -m spacy download en_core_web_sm

# 7. 驗證安裝
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
```

### 方法二：使用conda

```bash
# 1. 建立conda環境
conda create -n sentiment_analysis python=3.9
conda activate sentiment_analysis

# 2. 安裝PyTorch（根據您的CUDA版本）
# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# GPU版本（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安裝其他依賴
pip install transformers datasets scikit-learn pandas numpy matplotlib seaborn
pip install fastapi uvicorn nltk spacy beautifulsoup4 lime wandb

# 4. 下載語言模型
python -m spacy download en_core_web_sm
```

## 📊 資料準備

### 自動下載IMDB資料集

```python
from src.data import IMDBDataLoader

# 創建資料載入器
data_loader = IMDBDataLoader(cache_dir="data/raw")

# 自動下載和載入資料
train_texts, train_labels, test_texts, test_labels = data_loader.load_data()

print(f"訓練集: {len(train_texts)} 筆")
print(f"測試集: {len(test_texts)} 筆")
```

### 手動資料準備（可選）

如果自動下載失敗，可以手動準備資料：

```bash
# 1. 建立資料目錄
mkdir -p data/raw data/processed

# 2. 下載IMDB資料集
# 請從 https://ai.stanford.edu/~amaas/data/sentiment/ 下載
# 或使用kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# 3. 解壓到 data/raw/ 目錄
```

## 🚀 模型訓練

### 傳統機器學習模型

```bash
# 啟動Jupyter Notebook
jupyter lab notebooks/02_baseline_models.ipynb
```

或使用Python腳本：

```python
from src.models import BaselineModelManager
from src.data import IMDBDataLoader, TextPreprocessor

# 載入和預處理資料
data_loader = IMDBDataLoader()
train_texts, train_labels, test_texts, test_labels = data_loader.load_data()

preprocessor = TextPreprocessor(
    remove_html=True,
    lowercase=True,
    handle_negations=True
)
train_texts_clean = preprocessor.preprocess_batch(train_texts)

# 訓練模型
model_manager = BaselineModelManager()
results = model_manager.train_all_models(train_texts_clean, train_labels)

# 評估模型
test_results = model_manager.evaluate_all_models(
    preprocessor.preprocess_batch(test_texts), 
    test_labels
)
```

### 深度學習模型

```python
from src.models import TextCNN, BiLSTM, DeepLearningModelManager
from src.training import DeepLearningTrainer

# 建立詞彙表和數據載入器
# ... (資料準備代碼)

# 創建模型
model_manager = DeepLearningModelManager()
model = model_manager.create_model('textcnn', vocab_size=10000)

# 設置訓練器
trainer = DeepLearningTrainer(model, train_loader, val_loader)
trainer.setup_optimizer_and_scheduler(learning_rate=1e-3)

# 訓練模型
history = trainer.train(epochs=10, early_stopping_patience=3)
```

### Transformer模型

```python
from src.models import DistilBERTClassifier, TransformerTrainer

# 創建模型
model = DistilBERTClassifier(num_labels=2)

# 設置訓練器
trainer = TransformerTrainer(model, train_loader, val_loader)
trainer.setup_optimizer_and_scheduler(learning_rate=2e-5, num_epochs=3)

# 訓練模型
history = trainer.train(epochs=3, model_name='distilbert_imdb')
```

## 🌐 API服務

### 啟動服務

```bash
# 基本啟動
python app.py

# 指定埠號和主機
uvicorn src.inference.api:app --host 0.0.0.0 --port 8080 --reload

# 生產環境啟動
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### API端點說明

#### 1. 健康檢查
```http
GET /health
```

#### 2. 單個預測
```http
POST /predict
Content-Type: application/json

{
  "text": "This movie was fantastic!",
  "model_name": "distilbert_imdb"  // 可選
}
```

#### 3. 批次預測
```http
POST /predict/batch
Content-Type: application/json

{
  "texts": ["Great movie!", "Terrible film"],
  "model_name": "distilbert_imdb"  // 可選
}
```

#### 4. 模型管理
```http
GET /models                    // 獲取模型列表
POST /models/{model_name}/load // 載入指定模型
```

### API客戶端範例

```python
import requests
import json

# API基礎URL
BASE_URL = "http://localhost:8000"

# 測試健康檢查
response = requests.get(f"{BASE_URL}/health")
print(f"服務狀態: {response.json()['status']}")

# 單個預測
def predict_sentiment(text):
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    return response.json()

# 批次預測
def predict_batch(texts):
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": texts}
    )
    return response.json()

# 使用範例
result = predict_sentiment("Amazing movie with great acting!")
print(f"預測結果: {result['sentiment']} (信心度: {result['confidence']:.3f})")

batch_results = predict_batch([
    "Love this film!",
    "Worst movie ever",
    "It was okay"
])
for r in batch_results['results']:
    print(f"{r['text']} -> {r['sentiment']}")
```

## 🔧 進階配置

### 自定義預處理

```python
from src.data import TextPreprocessor

# 自定義預處理配置
preprocessor = TextPreprocessor(
    remove_html=True,
    remove_urls=True,
    lowercase=True,
    handle_negations=True,
    remove_stopwords=True,
    lemmatization=True,
    min_length=2
)

# 處理文本
processed_text = preprocessor.preprocess("This movie was <b>amazing</b>!")
```

### 模型配置

```python
# 深度學習模型配置
model_config = {
    'embed_dim': 128,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout_rate': 0.3
}

# Transformer模型配置
transformer_config = {
    'model_name': 'distilbert-base-uncased',
    'num_labels': 2,
    'max_length': 256,
    'freeze_base': False
}
```

### 訓練配置

```python
# 訓練超參數
training_config = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'epochs': 3,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'early_stopping_patience': 3
}
```

## ❓ 常見問題

### Q1: 為什麼模型訓練很慢？
**A**: 
- 確保使用GPU加速：檢查CUDA安裝和PyTorch GPU支援
- 減少batch_size：如果記憶體不足
- 使用較小的模型：如DistilBERT而非BERT
- 減少資料量：用於快速測試

### Q2: API服務啟動失敗？
**A**:
- 檢查埠號是否被佔用：`netstat -an | grep 8000`
- 確認模型檔案存在：檢查`experiments/models/`目錄
- 查看錯誤日誌：檢查終端輸出
- 確認依賴安裝：`pip list | grep fastapi`

### Q3: 記憶體不足錯誤？
**A**:
- 減少batch_size
- 使用gradient checkpointing
- 清理GPU記憶體：`torch.cuda.empty_cache()`
- 使用CPU模式：設置device='cpu'

### Q4: 模型準確率低？
**A**:
- 增加訓練epoch數
- 調整學習率
- 使用更大的模型
- 檢查資料品質和預處理

### Q5: 如何添加新的模型？
**A**:
```python
# 1. 在src/models/中定義新模型
class CustomModel(nn.Module):
    # 模型實作

# 2. 在ModelRegistry中註冊
registry.register_model(
    name="custom_model",
    model_path="path/to/model",
    model_type="pytorch"
)

# 3. 更新預測器支援
```

### Q6: 如何部署到生產環境？
**A**:
```bash
# 使用Docker
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api

# 使用Gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.inference.api:app

# 使用Nginx反向代理
# 配置nginx.conf指向FastAPI服務
```

## 📧 獲取支援

如果遇到問題：

1. 查看[GitHub Issues](issues)
2. 檢查日誌檔案
3. 提供完整的錯誤信息
4. 描述重現步驟

---

**✨ 祝您使用愉快！**