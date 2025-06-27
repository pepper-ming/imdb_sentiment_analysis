# 統計碩士深度學習專案：自然語言處理情感分析完整規劃

## 專案概述

本專案為統計碩士背景學生設計的6週深度學習實戰專案，聚焦於使用IMDB Movie Reviews資料集進行情感分析。專案涵蓋從理論基礎到實際部署的完整流程，結合統計學背景和現代NLP技術，提供漸進式學習路徑。

---

## 1. 專案企劃書和規格書

### 1.1 專案目標

**主要目標**：
- 構建高效能的電影評論情感分析系統，準確率達90%以上
- 掌握從傳統機器學習到深度學習的完整技術棧
- 建立可重現、可擴展的NLP專案開發流程
- 深入理解情感分析的理論基礎和實際應用

**學習目標**：
- 理解自然語言處理核心概念和數學基礎
- 掌握深度學習在情感分析中的應用
- 熟練使用PyTorch和Hugging Face Transformers
- 建立完整的MLOps工作流程

### 1.2 專案規格

**技術規格**：
- **程式語言**：Python 3.8+
- **主要框架**：PyTorch, Hugging Face Transformers
- **資料集**：IMDB Movie Reviews (50,000筆評論)
- **目標指標**：準確率 ≥ 90%, F1-Score ≥ 0.90
- **部署需求**：支援RESTful API推理服務

**功能規格**：
- 支援二元情感分類（正面/負面）
- 提供模型可解釋性分析
- 支援批次處理和實時推理
- 包含完整的實驗追蹤和性能監控

### 1.3 專案交付物

1. **完整的程式碼專案**（GitHub repository）
2. **技術文檔**（理論基礎、實作細節、API文檔）
3. **實驗報告**（模型比較、性能分析、結果討論）
4. **部署系統**（Docker容器化、API服務）
5. **專案展示**（demo應用、可視化分析）

---

## 2. 完整的專案架構設計

### 2.1 目錄結構

```
imdb_sentiment_analysis/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docker-compose.yml
├── Dockerfile
│
├── data/                          # 資料目錄
│   ├── raw/                      # 原始資料
│   ├── processed/                # 預處理後資料
│   └── external/                 # 外部資料
│
├── notebooks/                    # Jupyter筆記本
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_deep_learning_models.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_results_analysis.ipynb
│
├── src/                          # 原始程式碼
│   ├── __init__.py
│   ├── data/                     # 資料處理模組
│   │   ├── __init__.py
│   │   ├── dataset.py           # 資料集類別
│   │   ├── preprocessing.py     # 預處理功能
│   │   └── augmentation.py      # 資料增強
│   │
│   ├── models/                   # 模型定義
│   │   ├── __init__.py
│   │   ├── baseline.py          # 傳統機器學習模型
│   │   ├── deep_learning.py     # 深度學習模型
│   │   ├── transformers.py      # Transformer模型
│   │   └── ensemble.py          # 集成方法
│   │
│   ├── training/                 # 訓練相關
│   │   ├── __init__.py
│   │   ├── trainer.py           # 訓練類別
│   │   ├── loss.py              # 損失函數
│   │   └── metrics.py           # 評估指標
│   │
│   ├── evaluation/               # 評估模組
│   │   ├── __init__.py
│   │   ├── evaluator.py         # 評估器
│   │   └── visualization.py     # 結果可視化
│   │
│   ├── inference/                # 推理服務
│   │   ├── __init__.py
│   │   ├── predictor.py         # 預測器
│   │   └── api.py               # API服務
│   │
│   └── utils/                    # 工具函數
│       ├── __init__.py
│       ├── config.py            # 配置管理
│       ├── logger.py            # 日誌系統
│       └── helpers.py           # 輔助函數
│
├── configs/                      # 配置檔案
│   ├── model_configs/           # 模型配置
│   ├── training_configs/        # 訓練配置
│   └── deployment_configs/      # 部署配置
│
├── experiments/                  # 實驗記錄
│   ├── logs/                    # 訓練日誌
│   ├── models/                  # 儲存的模型
│   └── results/                 # 實驗結果
│
├── tests/                        # 測試程式碼
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
├── docs/                         # 文檔
│   ├── theory/                  # 理論基礎
│   ├── implementation/          # 實作指南
│   └── api/                     # API文檔
│
└── deployment/                   # 部署檔案
    ├── kubernetes/              # K8s配置
    ├── docker/                  # Docker檔案
    └── scripts/                 # 部署腳本
```

### 2.2 核心模組設計

**資料處理模組**：
- `dataset.py`：PyTorch Dataset類別，支援IMDB資料載入和預處理
- `preprocessing.py`：文本清理、分詞、向量化功能
- `augmentation.py`：資料增強技術（回翻譯、同義詞替換）

**模型架構模組**：
- `baseline.py`：傳統ML模型（SVM、邏輯回歸、樸素貝葉斯）
- `deep_learning.py`：RNN、LSTM、GRU、CNN實作
- `transformers.py`：BERT、RoBERTa、DistilBERT微調
- `ensemble.py`：模型集成和融合策略

**訓練和評估模組**：
- `trainer.py`：統一的訓練接口，支援多種模型類型
- `evaluator.py`：全面的模型評估，包含準確率、F1、AUC等指標
- `metrics.py`：自定義評估指標和可視化

---

## 3. 理論基礎深度解析

### 3.1 自然語言處理理論基礎

**文本向量化的數學基礎**：
自然語言處理的核心是將離散文本轉換為連續向量空間，基於分佈語意假設：相似語境中的詞彙具有相似語義。

**詞嵌入技術**：
- **Word2Vec**: 基於Skip-gram模型，目標函數為最大化條件概率
  ```
  L = Σᵢ Σⱼ∈context(i) log P(wⱼ|wᵢ)
  其中 P(wⱼ|wᵢ) = exp(vⱼᵀvᵢ) / Σₖ exp(vₖᵀvᵢ)
  ```

- **GloVe**: 結合全域和局部統計信息
  ```
  J = Σᵢ,ⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
  ```

- **FastText**: 考慮子詞信息，提升對未登錄詞的處理能力

### 3.2 深度學習理論基礎

**循環神經網路數學模型**：
基本RNN存在梯度消失問題，LSTM通過門控機制解決：

```
忘記門：fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
輸入門：iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
細胞狀態：Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ tanh(WC·[hₜ₋₁, xₜ] + bC)
輸出門：oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
隱藏狀態：hₜ = oₜ ⊙ tanh(Cₜ)
```

**Transformer自注意力機制**：
```
Attention(Q, K, V) = softmax(QKᵀ/√dk)V
```
其中注意力權重計算為：αᵢⱼ = exp(score(qᵢ, kⱼ)) / Σₖ exp(score(qᵢ, kₖ))

### 3.3 情感分析理論基礎

**統計學習角度**：
情感分析本質上是文本分類問題，可建模為：
- **二元分類**：P(y=1|x) = 1/(1 + exp(-wᵀx - b))
- **多類分類**：P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)

**貝葉斯方法**：樸素貝葉斯分類器假設特徵條件獨立
```
P(c|d) = P(d|c)P(c) / P(d) = Πᵢ P(wᵢ|c) · P(c) / P(d)
```

---

## 4. 技術堆疊和工具選擇

### 4.1 核心技術堆疊

**深度學習框架**：
- **PyTorch**: 動態計算圖，研究友好，與Hugging Face整合佳
- **Hugging Face Transformers**: 提供50萬+預訓練模型，簡化微調流程

**文本預處理工具**：
- **NLTK**: 基礎NLP功能（分詞、停用詞）
- **spaCy**: 高效能語言處理（詞形還原、POS標記）
- **Transformers Tokenizers**: 深度學習模型專用分詞器

**機器學習工具**：
- **scikit-learn**: 傳統ML算法和評估指標
- **pandas**: 資料操作和分析
- **numpy**: 數值計算基礎

### 4.2 開發環境配置

**推薦開發環境**：
```bash
# 創建虛擬環境
conda create -n sentiment_analysis python=3.9
conda activate sentiment_analysis

# 安裝核心套件
pip install torch transformers datasets accelerate
pip install scikit-learn pandas numpy matplotlib seaborn
pip install nltk spacy jupyterlab wandb

# 下載語言模型
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 4.3 特徵工程策略

**分層次特徵提取**：
1. **傳統方法**：TF-IDF向量化作為基線
2. **詞嵌入**：Word2Vec/GloVe用於語義理解
3. **深度特徵**：BERT embeddings提供上下文感知表示

**實作範例**：
```python
# TF-IDF特徵
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

# BERT特徵
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
```

---

## 5. IMDB資料集特性分析和處理方法

### 5.1 資料集特性深度分析

**規模和分佈**：
- 總量：50,000筆電影評論（25,000訓練 + 25,000測試）
- 標籤平衡：正面和負面評論各50%
- 高品質標準：負面評論≤4分，正面評論≥7分
- 電影多樣性：每部電影最多30條評論，避免單一電影偏見

**文本特徵統計**：
- 詞彙量：約124,252個唯一詞彙
- 常用詞限制：通常使用前10,000個詞彙
- 長度分佈：評論長度差異極大，需要padding處理
- 語言特點：包含HTML標籤、俚語、縮寫等真實網路文本

### 5.2 預處理pipeline設計

**HTML清理和標準化**：
```python
from bs4 import BeautifulSoup
import re

def preprocess_text(text):
    # 移除HTML標籤
    text = BeautifulSoup(text, 'html.parser').get_text()
    # 移除URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # 標準化空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 轉小寫
    text = text.lower()
    return text
```

**否定詞處理策略**：
```python
def handle_negations(text):
    # 重要：保留否定詞，因為它們對情感分析至關重要
    negations = ["not", "no", "never", "neither", "nowhere", "nothing"]
    words = text.split()
    for i, word in enumerate(words):
        if word in negations and i < len(words) - 1:
            words[i+1] = "NOT_" + words[i+1]
    return " ".join(words)
```

**長文本處理**：
- 最大長度：256-512個token（基於模型類型）
- 截斷策略：保留前256個token，通常包含最重要信息
- 填充策略：短文本用[PAD] token填充到統一長度

### 5.3 資料分割和驗證策略

**分割方案**：
- 使用官方訓練/測試分割（25k/25k）
- 從訓練集劃分20%作為驗證集（20k訓練/5k驗證/25k測試）
- 採用分層抽樣確保各集合中正負樣本平衡

**交叉驗證設計**：
- 5折交叉驗證評估模型穩定性
- 確保同一電影的評論不會同時出現在訓練和驗證集
- 使用種子固定保證結果可重現

---

## 6. 從傳統方法到先進方法的模型選擇

### 6.1 漸進式模型選擇策略

**第一階段：傳統機器學習（Week 1-2）**

**基線模型比較**：
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(kernel='rbf'),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}
```

**性能基準**：
- TF-IDF + 邏輯回歸：75-80%準確率
- TF-IDF + SVM：76-85%準確率
- TF-IDF + 樸素貝葉斯：68-75%準確率

### 6.2 深度學習模型進階（Week 3-4）

**CNN架構設計**：
```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) 
                  for conv_out in conv_outs]
        concat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(concat))
```

**LSTM/GRU架構**：
```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)
    
    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embed)
        # 使用最後一個時步的輸出
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(final_hidden))
```

### 6.3 Transformer模型（Week 4-5）

**DistilBERT微調**：
```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class DistilBERTClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
    
    def prepare_data(self, texts, labels, max_len=256):
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, 
            max_length=max_len, return_tensors='pt'
        )
        return Dataset(encodings, labels)
```

**性能比較表**：

| 模型類別 | 準確率 | F1-Score | 訓練時間 | 推理時間 | 參數量 |
|---------|--------|----------|----------|----------|--------|
| 邏輯回歸 | 80% | 0.79 | 2分鐘 | \<1ms | \<1M |
| SVM | 82% | 0.81 | 15分鐘 | 2ms | \<1M |
| CNN | 85% | 0.84 | 30分鐘 | 5ms | 2M |
| BiLSTM | 87% | 0.86 | 45分鐘 | 10ms | 3M |
| DistilBERT | 91% | 0.90 | 2小時 | 20ms | 66M |
| RoBERTa | 93% | 0.92 | 6小時 | 50ms | 125M |

---

## 7. 評估指標和實驗設計

### 7.1 全面評估指標體系

**基礎分類指標**：
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def comprehensive_evaluation(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_recall_fscore_support(y_true, y_pred, average='weighted')[0],
        'recall': precision_recall_fscore_support(y_true, y_pred, average='weighted')[1],
        'f1_score': precision_recall_fscore_support(y_true, y_pred, average='weighted')[2],
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
    
    return metrics
```

**進階評估指標**：
- **類別平衡指標**：精確率、召回率、F1分數（macro和weighted平均）
- **機率評估**：AUC-ROC、AUC-PR曲線
- **校準指標**：Brier Score、可靠性圖表
- **魯棒性測試**：對抗樣本、噪聲數據測試

### 7.2 實驗設計框架

**A/B測試設計**：
```python
import wandb

# 實驗追蹤配置
wandb.init(project="imdb-sentiment-analysis")

def run_experiment(model_config, data_config, training_config):
    wandb.config.update({
        **model_config,
        **data_config, 
        **training_config
    })
    
    # 訓練和評估
    model = create_model(model_config)
    results = train_and_evaluate(model, data_config, training_config)
    
    # 記錄結果
    wandb.log(results)
    return results
```

**統計顯著性測試**：
```python
from scipy.stats import ttest_rel

def compare_models(model1_scores, model2_scores):
    """比較兩個模型的性能差異是否顯著"""
    t_stat, p_value = ttest_rel(model1_scores, model2_scores)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 7.3 可解釋性分析

**注意力權重可視化**：
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(tokens, attention_weights, title="Attention Visualization"):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, 
                annot=True, fmt='.2f', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig
```

**LIME解釋性分析**：
```python
from lime.lime_text import LimeTextExplainer

def explain_prediction(model, text, num_features=10):
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    explanation = explainer.explain_instance(
        text, model.predict_proba, num_features=num_features
    )
    return explanation
```

---

## 8. 6週開發時程規劃

### 週次1-2：基礎建設和傳統方法

**Week 1: 環境建置和資料探索**
- **Day 1-2**: 開發環境設置、專案結構建立
- **Day 3-4**: IMDB資料集探索性分析
- **Day 5-7**: 資料預處理pipeline開發

**Week 2: 傳統機器學習基線**
- **Day 1-3**: TF-IDF特徵工程和基線模型實作
- **Day 4-5**: 模型比較和超參數調優
- **Day 6-7**: 結果分析和文檔撰寫

### 週次3-4：深度學習模型

**Week 3: CNN和RNN模型**
- **Day 1-2**: 詞嵌入（Word2Vec/GloVe）整合
- **Day 3-4**: CNN模型設計和訓練
- **Day 5-7**: LSTM/GRU模型實作和比較

**Week 4: Transformer入門**
- **Day 1-3**: BERT理論學習和DistilBERT微調
- **Day 4-5**: 不同Transformer模型比較
- **Day 6-7**: 模型集成和混合架構探索

### 週次5-6：優化和部署

**Week 5: 高級優化**
- **Day 1-2**: 超參數優化和模型調優
- **Day 3-4**: 資料增強和正則化技術
- **Day 5-7**: 可解釋性分析和實驗設計

**Week 6: 部署和總結**
- **Day 1-3**: API服務開發和容器化部署
- **Day 4-5**: 完整實驗報告撰寫
- **Day 6-7**: 專案展示準備和成果整理

### 詳細時程表

| 週次 | 任務 | 主要交付物 | 預期成果 |
|------|------|------------|----------|
| W1 | 環境建置 + 資料分析 | EDA報告、預處理pipeline | 理解資料特性 |
| W2 | 傳統ML基線 | 基線模型代碼、性能報告 | 80%+準確率基線 |
| W3 | 深度學習模型 | CNN/RNN實作、比較分析 | 85%+準確率 |
| W4 | Transformer模型 | BERT微調、模型比較 | 90%+準確率 |
| W5 | 優化和集成 | 優化策略、集成模型 | 92%+準確率 |
| W6 | 部署和總結 | API服務、完整報告 | 可用系統 |

---

## 9. 程式碼架構範例

### 9.1 資料處理架構

```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='distilbert-base-uncased', max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```

### 9.2 模型訓練架構

```python
# src/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb

class SentimentTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 資料載入器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False
        )
        
        # 優化器和調度器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        total_steps = len(self.train_loader) * config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # 損失函數
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return {'accuracy': accuracy, 'loss': avg_loss}
    
    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            # 記錄到wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            })
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
```

### 9.3 推理服務架構

```python
# src/inference/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI(title="IMDB Sentiment Analysis API")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class SentimentPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=256
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = "positive" if predicted_class == 1 else "negative"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence
        }

# 全域預測器
predictor = SentimentPredictor("./models/best_model")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    try:
        result = predictor.predict(request.text)
        return PredictionResponse(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 10. 最新的NLP情感分析技術趨勢和Best Practices

### 10.1 2024-2025年技術趨勢

**大型語言模型的情感分析應用**：
- **GPT-4系列**：在zero-shot和few-shot情感分析中表現卓越
- **實際案例**：Microsoft Bing Chat整合GPT-4進行實時情感分析
- **多語言支援**：在低資源語言情感分析中展現強大能力
- **提示工程優化**：26項系統性原則提升LLM情感分析效果

**輕量化模型發展**：
- **MobileBERT**：比BERT-base小4.3倍、快5.5倍，適合邊緣計算
- **TinyBERT**：通過知識蒸餾技術，將BERT壓縮到14M參數
- **量化技術**：INT8量化可將模型大小減少75%，推理速度提升2-4倍

**多模態情感分析**：
- **跨模態注意力**：整合文本和圖像信息進行情感分析
- **美學感知融合**：Atlantis框架的多粒度融合網絡
- **實際應用**：社交媒體監控、客戶反饋分析

### 10.2 提示工程最佳實踐

**情感分析專用提示策略**：
```python
def create_sentiment_prompt(text, style="simple"):
    if style == "simple":
        return f"分析以下文本的情感，回答正面、負面或中性：\n文本：{text}\n情感："
    
    elif style == "chain_of_thought":
        return f"""
        請分析以下文本的情感，並說明你的推理過程：
        
        文本：{text}
        
        分析步驟：
        1. 識別關鍵情感詞彙
        2. 考慮上下文和語調
        3. 綜合判斷整體情感
        
        最終判斷：
        """
    
    elif style == "few_shot":
        return f"""
        以下是一些情感分析的例子：
        
        文本："這部電影太棒了！" → 正面
        文本："劇情很無聊，不推薦。" → 負面
        文本："還可以，沒什麼特別的。" → 中性
        
        現在分析：
        文本：{text} → 
        """
```

### 10.3 模型部署最佳實踐

**容器化部署**：
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY src/ ./src/
COPY models/ ./models/

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes部署配置**：
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analysis
  template:
    metadata:
      labels:
        app: sentiment-analysis
    spec:
      containers:
      - name: api
        image: sentiment-analysis:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### 10.4 MLOps最佳實踐

**實驗追蹤和版本控制**：
```python
# 實驗配置管理
import wandb
from hydra import compose, initialize

def run_experiment_with_config():
    with initialize(config_path="../configs"):
        cfg = compose(config_name="experiment_config")
    
    wandb.init(
        project="imdb-sentiment",
        name=cfg.experiment.name,
        config=cfg
    )
    
    # 模型訓練
    trainer = SentimentTrainer(cfg)
    results = trainer.train()
    
    # 模型註冊
    wandb.log_model(
        path="./models/best_model",
        name="sentiment_classifier",
        aliases=["production", f"v{cfg.experiment.version}"]
    )
    
    return results
```

**持續集成pipeline**：
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/
    
    - name: Run model validation
      run: python scripts/validate_model.py
```

### 10.5 性能優化技巧

**模型量化和剪枝**：
```python
from transformers import AutoModelForSequenceClassification
import torch

def quantize_model(model_path, output_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 動態量化
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 儲存量化模型
    torch.save(quantized_model.state_dict(), output_path)
    return quantized_model
```

**批次推理優化**：
```python
def batch_predict(texts, model, tokenizer, batch_size=32):
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=256
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.extend(batch_predictions.cpu().numpy())
    
    return predictions
```

### 10.6 可解釋性和公平性

**偏見檢測框架**：
```python
def detect_bias(model, test_data, sensitive_attributes):
    results = {}
    
    for attr in sensitive_attributes:
        # 按敏感屬性分組
        groups = test_data.groupby(attr)
        
        for group_name, group_data in groups:
            predictions = model.predict(group_data['text'])
            accuracy = accuracy_score(group_data['labels'], predictions)
            
            results[f"{attr}_{group_name}"] = {
                'accuracy': accuracy,
                'size': len(group_data),
                'positive_rate': (predictions == 1).mean()
            }
    
    return results
```

**注意力權重可視化**：
```python
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

def visualize_attention_weights(text, model, tokenizer, layer=-1, head=0):
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention = outputs.attentions[layer][0, head].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 創建熱力圖
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, 
                annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.tight_layout()
    
    return fig
```

---

## 專案成功指標與預期成果

### 技術成果指標
- **模型性能**：IMDB測試集上達到90%+準確率
- **推理效能**：API響應時間\<100ms
- **系統穩定性**：99%+服務可用性
- **程式碼品質**：80%+測試覆蓋率

### 學習成果指標
- **理論掌握**：深入理解NLP和深度學習核心概念
- **實作能力**：獨立完成端到端NLP專案
- **工程能力**：建立完整MLOps工作流程
- **創新能力**：提出改進方案和最佳化策略

### 專案交付清單
1. **GitHub Repository**：完整程式碼和文檔
2. **技術報告**：30-50頁詳細分析報告
3. **API服務**：容器化的推理服務
4. **Demo應用**：網頁界面展示系統功能
5. **簡報材料**：專案成果展示簡報

---

## 結語

本專案規劃為統計碩士背景學生提供了完整的深度學習情感分析學習路徑，從理論基礎到實際應用，從傳統方法到最新技術，涵蓋了現代NLP專案的各個方面。透過6週的系統性學習和實作，學生將獲得在NLP領域的核心競爭力，為未來的學術研究或工業應用奠定堅實基礎。

專案的設計考慮了統計學背景的優勢，強調數學理論與實際應用的結合，通過漸進式的學習曲線和完整的工程實踐，確保學生能夠深入理解情感分析的核心概念，並具備獨立開發和部署NLP系統的能力。