# IMDB 情感分析模型訓練指南

## 概述

本指南將引導您完成 IMDB 情感分析專案的模型訓練流程，包括傳統機器學習模型、深度學習模型、Transformer 模型以及集成模型的訓練方法。

---

## 環境準備

### 1. Python 環境設置

```bash
# 建議使用 Python 3.8+ 版本
python --version

# 安裝依賴套件
pip install -r requirements.txt

# 或手動安裝主要套件
pip install torch transformers datasets accelerate
pip install scikit-learn pandas numpy matplotlib seaborn
pip install nltk spacy jupyterlab wandb fastapi uvicorn
```

### 2. 下載必要的語言模型

```bash
# 下載 spaCy 英文模型
python -m spacy download en_core_web_sm

# 下載 NLTK 數據
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 資料準備

### 1. 下載 IMDB 資料集

確保 `data/raw/IMDB_Dataset.csv` 檔案存在。如果沒有，可以從以下方式獲取：

```bash
# 方法1: 手動下載
# 從 Kaggle 下載 IMDB Dataset
# 網址: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-movie-reviews

# 方法2: 使用程式下載 (需要 kaggle API)
# kaggle datasets download -d lakshmi25npathi/imdb-movie-reviews
```

### 2. 執行資料預處理

```bash
# 執行資料探索和預處理
python -c "
from src.data.preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
preprocessor.load_data('data/raw/IMDB_Dataset.csv')
preprocessor.preprocess_all()
preprocessor.save_processed_data('data/processed/preprocessed_data.pkl')
print('資料預處理完成!')
"
```

---

## 模型訓練流程

### 階段 1: 基線模型訓練

基線模型包括邏輯回歸、SVM、樸素貝葉斯和隨機森林。

```bash
# 方法1: 使用專用腳本
python train_models.py --model baseline --config configs/baseline_config.json

# 方法2: 使用 Python 程式碼
python -c "
import pickle
from src.models.baseline import BaselineModels
from src.utils.logger import get_logger

# 載入資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 訓練基線模型
baseline = BaselineModels()
baseline.fit(data['X_train_tfidf'], data['y_train'])

# 評估模型
results = baseline.evaluate(data['X_test_tfidf'], data['y_test'])
print('基線模型訓練完成!')
print(f'結果: {results}')

# 保存模型
baseline.save_models('experiments/models/baseline/')
"
```

**預期結果：**
- 邏輯回歸：~80-85% 準確率
- SVM：~82-87% 準確率
- 樸素貝葉斯：~75-80% 準確率
- 隨機森林：~83-88% 準確率

### 階段 2: 深度學習模型訓練

#### 2.1 LSTM/GRU 模型

```bash
# 訓練 BiLSTM 模型
python -c "
import pickle
import torch
from src.models.deep_learning import BiLSTMClassifier
from src.training.trainer import SentimentTrainer
from src.data.dataset import IMDBDataset

# 載入資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 準備訓練配置
config = {
    'vocab_size': 10000,
    'embed_dim': 128,
    'hidden_dim': 64,
    'num_layers': 2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 10,
    'weight_decay': 1e-5
}

# 創建模型
model = BiLSTMClassifier(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers']
)

# 準備數據集
train_dataset = IMDBDataset(data['texts_train'], data['y_train'])
val_dataset = IMDBDataset(data['texts_test'], data['y_test'])

# 訓練模型
trainer = SentimentTrainer(model, train_dataset, val_dataset, config)
trainer.train()

print('LSTM 模型訓練完成!')
"
```

#### 2.2 CNN 模型

```bash
# 訓練 TextCNN 模型
python -c "
import pickle
import torch
from src.models.deep_learning import TextCNN
from src.training.trainer import SentimentTrainer

# 載入資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 訓練配置
config = {
    'vocab_size': 10000,
    'embed_dim': 128,
    'num_filters': 100,
    'filter_sizes': [3, 4, 5],
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 10
}

# 創建和訓練模型
model = TextCNN(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    num_filters=config['num_filters'],
    filter_sizes=config['filter_sizes']
)

# 訓練流程同上...
print('CNN 模型訓練完成!')
"
```

**預期結果：**
- BiLSTM：~87-90% 準確率
- TextCNN：~85-88% 準確率

### 階段 3: Transformer 模型訓練

#### 3.1 DistilBERT 微調

```bash
# 安裝 transformers 套件
pip install transformers datasets accelerate

# 訓練 DistilBERT
python -c "
import pickle
from src.models.transformers import TransformerClassifier
from transformers import TrainingArguments, Trainer

# 載入資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 創建 Transformer 分類器
classifier = TransformerClassifier(
    model_name='distilbert-base-uncased',
    num_labels=2
)

# 準備訓練參數
training_args = TrainingArguments(
    output_dir='./experiments/models/distilbert',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_dir='./experiments/logs',
)

# 開始訓練
classifier.train(
    train_texts=data['texts_train'][:5000],  # 可調整樣本數量
    train_labels=data['y_train'][:5000],
    val_texts=data['texts_test'][:1000],
    val_labels=data['y_test'][:1000],
    training_args=training_args
)

print('DistilBERT 模型訓練完成!')
"
```

#### 3.2 其他 Transformer 模型

```bash
# RoBERTa 模型
python -c "
from src.models.transformers import TransformerClassifier

classifier = TransformerClassifier('roberta-base', num_labels=2)
# 訓練流程同上...
"

# BERT 模型
python -c "
from src.models.transformers import TransformerClassifier

classifier = TransformerClassifier('bert-base-uncased', num_labels=2)
# 訓練流程同上...
"
```

**預期結果：**
- DistilBERT：~91-93% 準確率
- RoBERTa：~92-94% 準確率
- BERT：~91-93% 準確率

### 階段 4: 集成模型訓練

```bash
# 訓練集成模型
python -c "
import pickle
from src.models.ensemble import EnsembleManager, VotingEnsemble, WeightedAverageEnsemble
from src.models.baseline import BaselineModels

# 載入資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 訓練基線模型作為基學習器
baseline = BaselineModels()
baseline.fit(data['X_train_tfidf'], data['y_train'])

# 獲取訓練好的模型
models = [
    baseline.models['logistic_regression'],
    baseline.models['svm'],
    baseline.models['random_forest']
]

# 創建集成器
soft_voting = VotingEnsemble(models, voting_type='soft')
weighted_ensemble = WeightedAverageEnsemble(models)

# 基於驗證集計算權重
from sklearn.model_selection import train_test_split
X_val, X_test, y_val, y_test = train_test_split(
    data['X_test_tfidf'], data['y_test'], test_size=0.5, random_state=42
)
weighted_ensemble.fit_weights(X_val, y_val)

# 評估集成模型
manager = EnsembleManager()
manager.add_ensemble('soft_voting', soft_voting)
manager.add_ensemble('weighted_average', weighted_ensemble)

results = manager.evaluate_ensembles(X_test, y_test)
print(f'集成模型結果: {results}')

# 保存最佳集成模型
best_ensemble = manager.get_best_ensemble()
manager.save_ensemble('best', 'experiments/models/best_ensemble.pkl')
print('集成模型訓練完成!')
"
```

**預期結果：**
- 軟投票：~88-92% 準確率
- 加權平均：~89-93% 準確率

---

## 高級訓練技巧

### 1. 超參數調優

```bash
# 使用 GridSearch 進行超參數調優
python -c "
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle

# 載入資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# 定義參數網格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# 執行網格搜索
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(data['X_train_tfidf'], data['y_train'])
print(f'最佳參數: {grid_search.best_params_}')
print(f'最佳分數: {grid_search.best_score_:.4f}')
"
```

### 2. 資料增強

```bash
# 使用回翻譯進行資料增強
python -c "
from src.data.augmentation import DataAugmenter

augmenter = DataAugmenter()
# 注意：需要設置翻譯API密鑰
augmented_texts = augmenter.back_translate(texts[:100], target_lang='de')
print('資料增強完成!')
"
```

### 3. 模型蒸餾

```bash
# 使用大模型作為教師模型蒸餾小模型
python -c "
from src.training.distillation import ModelDistillation

# 載入教師模型(BERT)和學生模型(BiLSTM)
teacher_model = # 載入訓練好的BERT模型
student_model = # 創建BiLSTM模型

distiller = ModelDistillation(teacher_model, student_model)
distiller.distill(train_data, temperature=4.0, alpha=0.7)
print('模型蒸餾完成!')
"
```

---

## 模型評估與選擇

### 1. 全面評估

```bash
# 執行完整的模型評估
python -c "
import pickle
from src.evaluation.evaluator import ModelEvaluator

# 載入測試資料
with open('data/processed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

evaluator = ModelEvaluator()

# 載入所有訓練好的模型進行比較
models = {
    'baseline': 'experiments/models/baseline/',
    'lstm': 'experiments/models/lstm.pth',
    'cnn': 'experiments/models/cnn.pth',
    'distilbert': 'experiments/models/distilbert/',
    'ensemble': 'experiments/models/best_ensemble.pkl'
}

# 評估所有模型
results = evaluator.compare_models(models, data['X_test_tfidf'], data['y_test'])
evaluator.generate_report(results, 'experiments/results/model_comparison.html')
print('模型評估完成!')
"
```

### 2. 性能基準

| 模型類別 | 預期準確率 | 訓練時間 | 推理時間 | 模型大小 |
|---------|-----------|----------|----------|----------|
| 邏輯回歸 | 80-85% | 2分鐘 | <1ms | <1MB |
| SVM | 82-87% | 10分鐘 | 2ms | <1MB |
| BiLSTM | 87-90% | 30分鐘 | 10ms | 5MB |
| TextCNN | 85-88% | 20分鐘 | 5ms | 3MB |
| DistilBERT | 91-93% | 2小時 | 20ms | 250MB |
| RoBERTa | 92-94% | 6小時 | 50ms | 500MB |
| 集成模型 | 89-93% | 變動 | 變動 | 變動 |

---

## 模型部署準備

### 1. 模型轉換和優化

```bash
# 將 PyTorch 模型轉換為 ONNX 格式
python -c "
import torch
import torch.onnx
from src.models.deep_learning import BiLSTMClassifier

# 載入模型
model = BiLSTMClassifier(vocab_size=10000, embed_dim=128, hidden_dim=64, num_layers=2)
model.load_state_dict(torch.load('experiments/models/lstm.pth'))
model.eval()

# 創建範例輸入
dummy_input = torch.randint(0, 10000, (1, 256))

# 轉換為 ONNX
torch.onnx.export(
    model,
    dummy_input,
    'experiments/models/lstm_model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
print('模型轉換完成!')
"
```

### 2. 模型量化

```bash
# 量化 Transformer 模型
python -c "
from transformers import AutoModelForSequenceClassification
import torch

# 載入模型
model = AutoModelForSequenceClassification.from_pretrained('experiments/models/distilbert')

# 動態量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), 'experiments/models/distilbert_quantized.pth')
print('模型量化完成!')
"
```

---

## 故障排除

### 常見問題

1. **記憶體不足錯誤**
   ```bash
   # 減少批次大小
   config['batch_size'] = 16  # 或更小
   
   # 使用梯度累積
   config['gradient_accumulation_steps'] = 2
   ```

2. **CUDA 錯誤**
   ```bash
   # 檢查 CUDA 可用性
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 強制使用 CPU
   device = torch.device('cpu')
   ```

3. **模型收斂問題**
   ```bash
   # 調整學習率
   config['learning_rate'] = 1e-4  # 降低學習率
   
   # 增加訓練輪數
   config['epochs'] = 20
   
   # 添加學習率調度器
   from torch.optim.lr_scheduler import StepLR
   scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
   ```

### 訓練監控

```bash
# 使用 Weights & Biases 進行實驗追蹤
pip install wandb
wandb login

# 在訓練腳本中添加
python -c "
import wandb

wandb.init(project='imdb-sentiment-analysis')
wandb.config.update(config)

# 在訓練循環中記錄指標
wandb.log({'train_loss': loss, 'val_accuracy': accuracy})
"
```

---

## 模型性能優化建議

1. **資料層面**
   - 增加訓練數據量
   - 改善資料品質和清理
   - 使用資料增強技術

2. **模型層面**
   - 嘗試不同的模型架構
   - 調整超參數
   - 使用預訓練模型

3. **訓練層面**
   - 使用適當的正則化
   - 實施早停機制
   - 調整學習率策略

4. **集成層面**
   - 結合多種模型類型
   - 使用堆疊(Stacking)方法
   - 實施動態權重調整

---

## 總結

按照本指南的步驟，您應該能夠：

1. ✅ 成功訓練基線模型（準確率 >80%）
2. ✅ 實現深度學習模型（準確率 >85%）
3. ✅ 微調 Transformer 模型（準確率 >90%）
4. ✅ 建立集成模型（準確率 >92%）
5. ✅ 完成模型評估和比較
6. ✅ 準備生產環境部署

如需更詳細的技術說明，請參考：
- `notebooks/` 目錄中的 Jupyter notebook 範例
- `src/` 目錄中的模組化程式碼
- `configs/` 目錄中的配置檔案範例