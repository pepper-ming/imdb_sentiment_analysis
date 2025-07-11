"""
推論模組

提供模型推論和API服務功能，包括：
- predictor: 情感預測器
- api: FastAPI REST服務

主要類別：
- SentimentPredictor: 情感預測器
- ModelRegistry: 模型註冊管理器
"""

from .predictor import SentimentPredictor, ModelRegistry

__all__ = [
    'SentimentPredictor',
    'ModelRegistry'
]