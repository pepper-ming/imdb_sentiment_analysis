# IMDB電影評論情感分析專案

## 專案概述
本專案使用深度學習技術對IMDB電影評論進行情感分析，判斷評論的正面或負面情感。

## 專案結構
```
imdb_sentiment_analysis/
├── data/
│   ├── raw/           # 原始數據
│   └── processed/     # 處理後的數據
├── src/               # 源代碼
├── models/            # 訓練好的模型
├── notebooks/         # Jupyter notebooks
├── results/           # 實驗結果
├── config/            # 配置文件
├── tests/             # 測試文件
└── requirements.txt   # 依賴包列表
```

## 技術棧
- Python 3.8+
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn

## 快速開始

### 1. 環境設置
```bash
# 自動安裝所有依賴並設置環境
python setup.py

# 或手動安裝
pip install -r requirements.txt
```

### 2. 使用方式

#### 方式一：使用主腳本 (推薦)
```bash
# 數據預處理
python main.py preprocess

# 訓練模型
python main.py train --model simple_lstm --epochs 5

# 預測文本
python main.py predict --text "這部電影真的很棒！"
```

#### 方式二：使用獨立腳本
```bash
# 預測單個文本
python predict.py --text "這部電影很無聊"

# 互動模式預測
python predict.py --interactive

# 使用指定模型
python predict.py --model models/simple_lstm_final.h5 --text "很棒的電影"
```

#### 方式三：使用Jupyter Notebook
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. 支持的模型類型
- `simple_lstm`: 簡單LSTM模型
- `bidirectional_lstm`: 雙向LSTM模型  
- `cnn`: 卷積神經網路模型
- `cnn_lstm`: CNN+LSTM混合模型