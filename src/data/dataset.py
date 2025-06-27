"""
IMDB資料集處理模組

提供IMDB電影評論資料集的載入、預處理和PyTorch Dataset實作。
支援傳統機器學習和深度學習模型的資料格式需求。
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class IMDBDataset(Dataset):
    """
    IMDB電影評論PyTorch Dataset類別
    
    支援多種tokenizer和預處理選項，適用於不同的模型架構。
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer_name: str = 'distilbert-base-uncased',
        max_length: int = 256,
        is_bert_like: bool = True
    ):
        """
        初始化IMDB資料集
        
        Args:
            texts: 評論文本列表
            labels: 情感標籤列表 (0: 負面, 1: 正面)
            tokenizer_name: tokenizer名稱
            max_length: 最大序列長度
            is_bert_like: 是否使用BERT類型的tokenizer
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.is_bert_like = is_bert_like
        
        if is_bert_like:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
    
    def __len__(self) -> int:
        """返回資料集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        獲取單個樣本
        
        Args:
            idx: 樣本索引
            
        Returns:
            包含輸入特徵和標籤的字典
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.is_bert_like and self.tokenizer:
            # 使用BERT類型的tokenizer
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
                'labels': torch.tensor(label, dtype=torch.long),
                'text': text
            }
        else:
            # 返回原始文本，供傳統ML模型使用
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }


class IMDBDataLoader:
    """
    IMDB資料載入和管理類別
    
    負責從datasets庫載入IMDB資料，並提供資料分割和統計功能。
    """
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        初始化資料載入器
        
        Args:
            cache_dir: 資料快取目錄
        """
        self.cache_dir = cache_dir
        self.train_texts = None
        self.train_labels = None
        self.test_texts = None
        self.test_labels = None
    
    def load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        載入IMDB資料集
        
        Returns:
            (train_texts, train_labels, test_texts, test_labels)
        """
        try:
            from datasets import load_dataset
            
            # 載入IMDB資料集
            dataset = load_dataset("imdb", cache_dir=self.cache_dir)
            
            # 提取訓練集
            self.train_texts = dataset['train']['text']
            self.train_labels = dataset['train']['label']
            
            # 提取測試集
            self.test_texts = dataset['test']['text']
            self.test_labels = dataset['test']['label']
            
            return self.train_texts, self.train_labels, self.test_texts, self.test_labels
            
        except ImportError:
            raise ImportError("請安裝datasets套件: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"載入IMDB資料集失敗: {e}")
    
    def create_train_val_split(
        self, 
        val_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        將訓練集分割為訓練集和驗證集
        
        Args:
            val_size: 驗證集比例
            random_state: 隨機種子
            
        Returns:
            (train_texts, train_labels, val_texts, val_labels)
        """
        if self.train_texts is None:
            raise ValueError("請先呼叫load_data()載入資料")
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.train_texts, 
            self.train_labels,
            test_size=val_size,
            stratify=self.train_labels,
            random_state=random_state
        )
        
        return train_texts, train_labels, val_texts, val_labels
    
    def get_data_statistics(self) -> dict:
        """
        獲取資料集統計信息
        
        Returns:
            包含統計信息的字典
        """
        if self.train_texts is None:
            raise ValueError("請先呼叫load_data()載入資料")
        
        # 文本長度統計
        train_lengths = [len(text.split()) for text in self.train_texts]
        test_lengths = [len(text.split()) for text in self.test_texts]
        
        # 標籤分佈
        train_positive = sum(self.train_labels)
        test_positive = sum(self.test_labels)
        
        return {
            'train_size': len(self.train_texts),
            'test_size': len(self.test_texts),
            'train_positive_ratio': train_positive / len(self.train_labels),
            'test_positive_ratio': test_positive / len(self.test_labels),
            'avg_train_length': np.mean(train_lengths),
            'avg_test_length': np.mean(test_lengths),
            'max_train_length': max(train_lengths),
            'max_test_length': max(test_lengths),
            'min_train_length': min(train_lengths),
            'min_test_length': min(test_lengths)
        }