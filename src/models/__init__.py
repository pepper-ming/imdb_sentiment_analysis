"""
模型模組

提供各種機器學習和深度學習模型的實作，包括：
- baseline: 傳統機器學習基線模型
- deep_learning: 深度學習模型 (CNN, RNN, LSTM)
- transformers: Transformer模型 (BERT, DistilBERT, RoBERTa)
- ensemble: 集成學習方法

主要類別：
- BaselineModelManager: 傳統ML模型管理器
- TextCNN: 卷積神經網路文本分類
- BiLSTM: 雙向長短期記憶網路
- GRUClassifier: 門控循環單元分類器
- DeepLearningModelManager: 深度學習模型管理器
"""

from .baseline import BaselineModelManager
from .deep_learning import TextCNN, BiLSTM, GRUClassifier, DeepLearningModelManager

__all__ = [
    'BaselineModelManager',
    'TextCNN',
    'BiLSTM', 
    'GRUClassifier',
    'DeepLearningModelManager'
]