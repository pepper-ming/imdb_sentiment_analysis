"""
訓練模組

提供各種模型的訓練功能，包括：
- trainer: 訓練器類別
- loss: 損失函數
- metrics: 評估指標

主要類別：
- DeepLearningTrainer: 深度學習訓練器
"""

from .trainer import DeepLearningTrainer

__all__ = [
    'DeepLearningTrainer'
]