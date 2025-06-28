"""
評估模組

提供模型評估相關功能，包括：
- 準確率計算
- 混淆矩陣生成
- 分類報告
- 模型比較

主要類別：
- ModelEvaluator: 模型評估器
"""

from .evaluator import ModelEvaluator

__all__ = [
    'ModelEvaluator'
]