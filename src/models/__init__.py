"""
模型模組

提供各種機器學習和深度學習模型的實作，包括：
- baseline: 傳統機器學習基線模型
- deep_learning: 深度學習模型 (CNN, RNN, LSTM)
- transformers: Transformer模型 (BERT, DistilBERT, RoBERTa)

主要類別：
- BaselineModelManager: 傳統ML模型管理器
- TextCNN: 卷積神經網路文本分類
- BiLSTM: 雙向長短期記憶網路
- GRUClassifier: 門控循環單元分類器
- DeepLearningModelManager: 深度學習模型管理器
- DistilBERTClassifier: DistilBERT分類器
- RoBERTaClassifier: RoBERTa分類器
- TransformerTrainer: Transformer訓練器
- TransformerModelManager: Transformer模型管理器
"""

from .baseline import BaselineModelManager
from .deep_learning import TextCNN, BiLSTM, GRUClassifier, DeepLearningModelManager
from .transformers import (
    DistilBERTClassifier, RoBERTaClassifier, 
    TransformerTrainer, TransformerModelManager
)

__all__ = [
    'BaselineModelManager',
    'TextCNN',
    'BiLSTM', 
    'GRUClassifier',
    'DeepLearningModelManager',
    'DistilBERTClassifier',
    'RoBERTaClassifier',
    'TransformerTrainer',
    'TransformerModelManager'
]