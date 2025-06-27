"""
Transformer模型模組

實作多種Transformer架構進行情感分析，包括：
- DistilBERT: 輕量化BERT模型
- RoBERTa: 強化版BERT
- BERT: 原始BERT模型
- 支援微調和特徵提取
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    BertForSequenceClassification, BertTokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os

from ..utils.logger import logger


class TransformerClassifier:
    """
    Transformer分類器基礎類別
    
    提供統一的Transformer模型接口。
    """
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 2,
        max_length: int = 256,
        device: str = 'cuda'
    ):
        """
        初始化Transformer分類器
        
        Args:
            model_name: 預訓練模型名稱
            num_labels: 分類標籤數
            max_length: 最大序列長度
            device: 計算設備
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        
        # 載入tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(device)
        
        logger.info(f"載入Transformer模型: {model_name}")
        logger.info(f"模型參數量: {self.count_parameters():,}")
    
    def count_parameters(self) -> int:
        """計算模型參數量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        對文本進行tokenization
        
        Args:
            texts: 文本列表
            
        Returns:
            tokenization結果字典
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {key: val.to(self.device) for key, val in encodings.items()}
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            input_ids: 輸入token IDs
            attention_mask: 注意力遮罩
            
        Returns:
            模型輸出
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測文本標籤
        
        Args:
            texts: 文本列表
            
        Returns:
            (預測標籤, 預測機率)
        """
        self.model.eval()
        
        # Tokenize
        encodings = self.tokenize_texts(texts)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            
            # 計算機率和預測
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions.cpu().numpy(), probs.cpu().numpy()


class DistilBERTClassifier(TransformerClassifier):
    """
    DistilBERT分類器
    
    輕量化BERT模型，保持97%性能但參數量減少40%。
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        max_length: int = 256,
        device: str = 'cuda',
        freeze_base: bool = False
    ):
        """
        初始化DistilBERT分類器
        
        Args:
            num_labels: 分類標籤數
            max_length: 最大序列長度
            device: 計算設備
            freeze_base: 是否凍結基礎層
        """
        super().__init__('distilbert-base-uncased', num_labels, max_length, device)
        
        if freeze_base:
            # 凍結DistilBERT基礎層
            for param in self.model.distilbert.parameters():
                param.requires_grad = False
            logger.info("DistilBERT基礎層已凍結")


class RoBERTaClassifier(TransformerClassifier):
    """
    RoBERTa分類器
    
    強化版BERT模型，移除Next Sentence Prediction並優化訓練策略。
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        max_length: int = 256,
        device: str = 'cuda',
        freeze_base: bool = False
    ):
        """
        初始化RoBERTa分類器
        
        Args:
            num_labels: 分類標籤數
            max_length: 最大序列長度
            device: 計算設備
            freeze_base: 是否凍結基礎層
        """
        super().__init__('roberta-base', num_labels, max_length, device)
        
        if freeze_base:
            # 凍結RoBERTa基礎層
            for param in self.model.roberta.parameters():
                param.requires_grad = False
            logger.info("RoBERTa基礎層已凍結")


class TransformerTrainer:
    """
    Transformer模型訓練器
    
    專門針對Transformer模型優化的訓練框架。
    """
    
    def __init__(
        self,
        model: TransformerClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = 'experiments/models'
    ):
        """
        初始化Transformer訓練器
        
        Args:
            model: Transformer分類器
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            output_dir: 模型輸出目錄
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.device = model.device
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # 最佳模型追蹤
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def setup_optimizer_and_scheduler(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1
    ):
        """
        設置優化器和學習率調度器
        
        Args:
            learning_rate: 學習率
            weight_decay: 權重衰減
            num_epochs: 訓練epoch數
            warmup_ratio: 預熱比例
        """
        # AdamW優化器（推薦用於Transformer）
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 計算總步數
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # 線性學習率調度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"優化器設置完成 - LR: {learning_rate}, 總步數: {total_steps}, 預熱步數: {warmup_steps}")
    
    def train_epoch(self) -> float:
        """
        訓練一個epoch
        
        Returns:
            平均訓練損失
        """
        self.model.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移動資料到設備
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            
            # 更新參數
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # 記錄進度
            if batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        驗證一個epoch
        
        Returns:
            (平均驗證損失, 驗證準確率)
        """
        self.model.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        epochs: int = 3,
        save_best_model: bool = True,
        model_name: str = 'transformer_best'
    ) -> Dict[str, List[float]]:
        """
        執行完整訓練流程
        
        Args:
            epochs: 訓練epoch數
            save_best_model: 是否保存最佳模型
            model_name: 模型名稱
            
        Returns:
            訓練歷史記錄
        """
        logger.info(f"開始訓練Transformer模型，共 {epochs} epochs")
        
        for epoch in range(epochs):
            # 訓練
            train_loss = self.train_epoch()
            
            # 驗證
            val_loss, val_accuracy = self.validate_epoch()
            
            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # 記錄結果
            logger.info(f'Epoch {epoch+1}/{epochs}')
            logger.info(f'Train Loss: {train_loss:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # 保存最佳模型
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                
                if save_best_model:
                    model_path = os.path.join(self.output_dir, f'{model_name}')
                    self.model.model.save_pretrained(model_path)
                    self.model.tokenizer.save_pretrained(model_path)
                    self.best_model_path = model_path
                    logger.info(f'保存最佳模型: {model_path}')
        
        logger.info(f'訓練完成，最佳驗證準確率: {self.best_val_acc:.4f}')
        
        return self.history
    
    def load_model(self, model_path: str):
        """
        載入訓練好的模型
        
        Args:
            model_path: 模型目錄路徑
        """
        self.model.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.model.to(self.device)
        logger.info(f'模型載入成功: {model_path}')


class TransformerModelManager:
    """
    Transformer模型管理器
    
    提供統一的Transformer模型創建和管理接口。
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化模型管理器
        
        Args:
            device: 計算設備
        """
        self.device = device
        
        # 支援的模型配置
        self.model_configs = {
            'distilbert': {
                'class': DistilBERTClassifier,
                'name': 'distilbert-base-uncased'
            },
            'roberta': {
                'class': RoBERTaClassifier,
                'name': 'roberta-base'
            }
        }
    
    def create_model(
        self,
        model_type: str,
        num_labels: int = 2,
        max_length: int = 256,
        **kwargs
    ) -> TransformerClassifier:
        """
        創建Transformer模型
        
        Args:
            model_type: 模型類型
            num_labels: 分類標籤數
            max_length: 最大序列長度
            **kwargs: 其他參數
            
        Returns:
            Transformer分類器實例
        """
        if model_type not in self.model_configs:
            raise ValueError(f"不支援的模型類型: {model_type}")
        
        model_class = self.model_configs[model_type]['class']
        
        model = model_class(
            num_labels=num_labels,
            max_length=max_length,
            device=self.device,
            **kwargs
        )
        
        return model