"""
資料處理模組

提供IMDB資料集的載入、預處理和管理功能：
- 資料載入和探索
- 文本預處理和清理
- PyTorch Dataset包裝
- 特徵工程和向量化

主要類別：
- IMDBDataset: PyTorch資料集類別
- IMDBDataLoader: 資料載入和管理
- TextPreprocessor: 文本預處理工具
"""

from .dataset import IMDBDataset, IMDBDataLoader
from .preprocessing import TextPreprocessor

__all__ = [
    'IMDBDataset',
    'IMDBDataLoader', 
    'TextPreprocessor'
]