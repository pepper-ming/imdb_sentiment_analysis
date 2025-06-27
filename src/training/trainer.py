"""
深度學習模型訓練模組

提供統一的深度學習模型訓練框架，支援：
- 自動混合精度訓練
- 學習率調度
- 早停法
- 梯度裁剪
- 訓練過程監控
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import time
import os
from collections import defaultdict

from ..utils.logger import logger


class DeepLearningTrainer:
    """
    深度學習訓練器
    
    提供完整的模型訓練、驗證和儲存功能。
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        output_dir: str = 'experiments/models'
    ):
        """
        初始化訓練器
        
        Args:
            model: PyTorch模型
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            device: 計算設備
            output_dir: 模型輸出目錄
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 訓練歷史記錄
        self.history = defaultdict(list)
        
        # 最佳模型追蹤
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        logger.info(f"訓練器初始化完成，使用設備: {device}")
    
    def setup_optimizer_and_scheduler(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = 'adam',
        scheduler_type: str = 'step',
        **scheduler_kwargs
    ):
        """
        設置優化器和學習率調度器
        
        Args:
            learning_rate: 學習率
            weight_decay: 權重衰減
            optimizer_type: 優化器類型 ('adam', 'sgd', 'adamw')
            scheduler_type: 調度器類型 ('step', 'cosine', 'plateau')
            **scheduler_kwargs: 調度器參數
        """
        # 設置優化器
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支援的優化器類型: {optimizer_type}")
        
        # 設置學習率調度器
        if scheduler_type.lower() == 'step':
            step_size = scheduler_kwargs.get('step_size', 10)
            gamma = scheduler_kwargs.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type.lower() == 'cosine':
            T_max = scheduler_kwargs.get('T_max', 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        elif scheduler_type.lower() == 'plateau':
            patience = scheduler_kwargs.get('patience', 5)
            factor = scheduler_kwargs.get('factor', 0.5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=patience, factor=factor
            )
        else:
            self.scheduler = None
        
        logger.info(f"優化器: {optimizer_type}, 調度器: {scheduler_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        訓練一個epoch
        
        Returns:
            epoch訓練結果字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移動資料到設備
            if isinstance(batch, dict):
                # 處理字典格式的batch (來自IMDBDataset)
                if 'text' in batch and 'labels' in batch:
                    # 傳統ML格式，跳過
                    continue
                else:
                    # BERT格式
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    inputs = input_ids
            else:
                # 處理tuple格式的batch
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 計算損失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新參數
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 記錄進度
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        驗證一個epoch
        
        Returns:
            epoch驗證結果字典
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移動資料到設備
                if isinstance(batch, dict):
                    if 'text' in batch and 'labels' in batch:
                        continue
                    else:
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        inputs = input_ids
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                # 前向傳播
                outputs = self.model(inputs)
                
                # 計算損失
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)
                
                # 統計
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        epochs: int = 10,
        early_stopping_patience: int = 5,
        save_best_model: bool = True,
        model_name: str = 'best_model'
    ) -> Dict[str, List[float]]:
        """
        執行完整訓練流程
        
        Args:
            epochs: 訓練epoch數
            early_stopping_patience: 早停耐心值
            save_best_model: 是否保存最佳模型
            model_name: 模型名稱
            
        Returns:
            訓練歷史記錄
        """
        logger.info(f"開始訓練，共 {epochs} epochs")
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 訓練
            train_results = self.train_epoch()
            
            # 驗證
            val_results = self.validate_epoch()
            
            # 更新學習率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['accuracy'])
                else:
                    self.scheduler.step()
            
            # 記錄歷史
            self.history['train_loss'].append(train_results['loss'])
            self.history['train_acc'].append(train_results['accuracy'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_acc'].append(val_results['accuracy'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 計算epoch時間
            epoch_time = time.time() - epoch_start_time
            
            # 記錄結果
            logger.info(f'Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)')
            logger.info(f'Train Loss: {train_results["loss"]:.4f}, '
                       f'Train Acc: {train_results["accuracy"]:.4f}')
            logger.info(f'Val Loss: {val_results["loss"]:.4f}, '
                       f'Val Acc: {val_results["accuracy"]:.4f}')
            logger.info(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 早停和模型保存
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                patience_counter = 0
                
                if save_best_model:
                    model_path = os.path.join(self.output_dir, f'{model_name}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_accuracy': best_val_acc,
                        'val_loss': val_results['loss']
                    }, model_path)
                    self.best_model_path = model_path
                    logger.info(f'保存最佳模型: {model_path}')
            else:
                patience_counter += 1
            
            # 早停檢查
            if patience_counter >= early_stopping_patience:
                logger.info(f'早停觸發，於第 {epoch+1} epoch')
                break
        
        total_time = time.time() - start_time
        logger.info(f'訓練完成，總時間: {total_time:.2f}s')
        logger.info(f'最佳驗證準確率: {best_val_acc:.4f}')
        
        return dict(self.history)
    
    def load_model(self, model_path: str):
        """
        載入訓練好的模型
        
        Args:
            model_path: 模型檔案路徑
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'模型載入成功: {model_path}')
    
    def predict(self, data_loader: DataLoader) -> tuple:
        """
        模型預測
        
        Args:
            data_loader: 資料載入器
            
        Returns:
            (predictions, probabilities)
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    if 'text' in batch and 'labels' in batch:
                        continue
                    else:
                        inputs = batch['input_ids'].to(self.device)
                else:
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)