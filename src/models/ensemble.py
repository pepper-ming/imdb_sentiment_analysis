"""
模型集成模組 - 實現多種集成策略以提升情感分析性能
支援投票、加權平均、Stacking等多種集成方法
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

from ..utils.logger import get_logger
from .baseline import BaselineModels
from .deep_learning import BiLSTMClassifier, TextCNN
from .transformers import TransformerClassifier

logger = get_logger(__name__)


class VotingEnsemble:
    """
    投票集成器 - 支援硬投票和軟投票
    """
    
    def __init__(self, models: List[Any], voting_type: str = 'soft', weights: Optional[List[float]] = None):
        """
        初始化投票集成器
        
        Args:
            models: 基學習器列表
            voting_type: 投票類型 ('hard' 或 'soft')
            weights: 模型權重，None時為等權重
        """
        self.models = models
        self.voting_type = voting_type
        self.weights = weights or [1.0] * len(models)
        self.num_models = len(models)
        
        if len(self.weights) != self.num_models:
            raise ValueError("權重數量必須與模型數量相等")
            
        # 正規化權重
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        
        logger.info(f"初始化{voting_type}投票集成器，包含{self.num_models}個模型")
    
    def predict(self, X: Any) -> np.ndarray:
        """
        集成預測
        
        Args:
            X: 輸入數據
            
        Returns:
            預測結果
        """
        if self.voting_type == 'hard':
            return self._hard_voting(X)
        else:
            return self._soft_voting(X)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        預測類別機率
        
        Args:
            X: 輸入數據
            
        Returns:
            類別機率分佈
        """
        predictions = []
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                # 對於深度學習模型，假設返回機率
                pred = model.predict(X)
                if len(pred.shape) == 1:
                    # 轉換為二分類機率
                    pred_proba = np.column_stack([1-pred, pred])
                else:
                    pred_proba = pred
            else:
                raise ValueError(f"模型 {i} 不支援機率預測")
            
            predictions.append(pred_proba * self.weights[i])
        
        return np.mean(predictions, axis=0)
    
    def _hard_voting(self, X: Any) -> np.ndarray:
        """硬投票實現"""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
            else:
                raise ValueError("模型必須支援predict方法")
        
        predictions = np.array(predictions)
        
        # 加權投票
        weighted_votes = []
        for i in range(predictions.shape[1]):
            votes = {}
            for j, vote in enumerate(predictions[:, i]):
                votes[vote] = votes.get(vote, 0) + self.weights[j]
            
            # 選擇得票最高的類別
            best_class = max(votes.keys(), key=lambda k: votes[k])
            weighted_votes.append(best_class)
        
        return np.array(weighted_votes)
    
    def _soft_voting(self, X: Any) -> np.ndarray:
        """軟投票實現"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class StackingEnsemble:
    """
    Stacking集成器 - 使用元學習器整合基學習器預測
    """
    
    def __init__(self, base_models: List[Any], meta_model: Any = None):
        """
        初始化Stacking集成器
        
        Args:
            base_models: 基學習器列表
            meta_model: 元學習器，默認為邏輯回歸
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=42)
        self.num_base_models = len(base_models)
        
        logger.info(f"初始化Stacking集成器，包含{self.num_base_models}個基學習器")
    
    def fit(self, X_train: Any, y_train: np.ndarray, X_val: Any, y_val: np.ndarray):
        """
        訓練Stacking集成器
        
        Args:
            X_train: 訓練數據
            y_train: 訓練標籤
            X_val: 驗證數據
            y_val: 驗證標籤
        """
        logger.info("開始訓練Stacking集成器...")
        
        # 獲取基學習器在驗證集上的預測作為元特徵
        meta_features = self._get_meta_features(X_val)
        
        # 訓練元學習器
        self.meta_model.fit(meta_features, y_val)
        
        logger.info("Stacking集成器訓練完成")
    
    def predict(self, X: Any) -> np.ndarray:
        """
        集成預測
        
        Args:
            X: 輸入數據
            
        Returns:
            預測結果
        """
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        預測類別機率
        
        Args:
            X: 輸入數據
            
        Returns:
            類別機率分佈
        """
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
    
    def _get_meta_features(self, X: Any) -> np.ndarray:
        """獲取元特徵"""
        meta_features = []
        
        for i, model in enumerate(self.base_models):
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)
                    meta_features.append(pred_proba[:, 1])  # 取正類機率
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                    if len(pred.shape) > 1:
                        pred = pred[:, 1] if pred.shape[1] > 1 else pred.flatten()
                    meta_features.append(pred)
                else:
                    raise ValueError(f"基模型 {i} 不支援預測")
            except Exception as e:
                logger.warning(f"基模型 {i} 預測失敗: {str(e)}")
                continue
        
        return np.column_stack(meta_features)


class WeightedAverageEnsemble:
    """
    加權平均集成器 - 基於驗證集性能自動計算權重
    """
    
    def __init__(self, models: List[Any], weight_strategy: str = 'accuracy'):
        """
        初始化加權平均集成器
        
        Args:
            models: 模型列表
            weight_strategy: 權重計算策略 ('accuracy', 'f1', 'manual')
        """
        self.models = models
        self.weight_strategy = weight_strategy
        self.weights = None
        self.num_models = len(models)
        
        logger.info(f"初始化加權平均集成器，權重策略: {weight_strategy}")
    
    def fit_weights(self, X_val: Any, y_val: np.ndarray, manual_weights: Optional[List[float]] = None):
        """
        根據驗證集性能計算權重
        
        Args:
            X_val: 驗證數據
            y_val: 驗證標籤
            manual_weights: 手動指定權重
        """
        if self.weight_strategy == 'manual' and manual_weights is not None:
            self.weights = np.array(manual_weights)
        else:
            scores = []
            
            for i, model in enumerate(self.models):
                try:
                    pred = model.predict(X_val)
                    if len(pred.shape) > 1:
                        pred = np.argmax(pred, axis=1)
                    
                    if self.weight_strategy == 'accuracy':
                        score = accuracy_score(y_val, pred)
                    else:  # f1
                        from sklearn.metrics import f1_score
                        score = f1_score(y_val, pred, average='weighted')
                    
                    scores.append(score)
                    logger.info(f"模型 {i} 在驗證集上的{self.weight_strategy}: {score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"模型 {i} 評估失敗: {str(e)}")
                    scores.append(0.0)
            
            # 計算權重 (性能越好權重越大)
            scores = np.array(scores)
            self.weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores)
        
        # 正規化權重
        self.weights = self.weights / np.sum(self.weights)
        
        logger.info(f"計算得到的權重: {self.weights}")
    
    def predict(self, X: Any) -> np.ndarray:
        """
        加權平均預測
        
        Args:
            X: 輸入數據
            
        Returns:
            預測結果
        """
        if self.weights is None:
            raise ValueError("請先調用fit_weights方法計算權重")
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            if len(pred.shape) > 1:
                pred = pred[:, 1] if pred.shape[1] > 1 else np.argmax(pred, axis=1)
            predictions.append(pred)
        
        # 加權平均
        weighted_pred = np.zeros(len(predictions[0]))
        for i, pred in enumerate(predictions):
            weighted_pred += pred * self.weights[i]
        
        # 轉換為類別預測
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        預測類別機率
        
        Args:
            X: 輸入數據
            
        Returns:
            類別機率分佈
        """
        if self.weights is None:
            raise ValueError("請先調用fit_weights方法計算權重")
        
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
            else:
                pred = model.predict(X)
                if len(pred.shape) == 1:
                    pred_proba = np.column_stack([1-pred, pred])
                else:
                    pred_proba = pred
            
            predictions.append(pred_proba)
        
        # 加權平均
        weighted_proba = np.zeros_like(predictions[0])
        for i, pred_proba in enumerate(predictions):
            weighted_proba += pred_proba * self.weights[i]
        
        return weighted_proba


class EnsembleManager:
    """
    集成管理器 - 統一管理多種集成策略
    """
    
    def __init__(self):
        """初始化集成管理器"""
        self.ensembles = {}
        self.best_ensemble = None
        self.best_score = 0.0
        
        logger.info("初始化集成管理器")
    
    def add_ensemble(self, name: str, ensemble: Any):
        """
        添加集成器
        
        Args:
            name: 集成器名稱
            ensemble: 集成器實例
        """
        self.ensembles[name] = ensemble
        logger.info(f"添加集成器: {name}")
    
    def evaluate_ensembles(self, X_test: Any, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        評估所有集成器性能
        
        Args:
            X_test: 測試數據
            y_test: 測試標籤
            
        Returns:
            評估結果字典
        """
        results = {}
        
        for name, ensemble in self.ensembles.items():
            try:
                logger.info(f"評估集成器: {name}")
                
                pred = ensemble.predict(X_test)
                if len(pred.shape) > 1:
                    pred = np.argmax(pred, axis=1)
                
                accuracy = accuracy_score(y_test, pred)
                
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average='weighted')
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # 更新最佳集成器
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_ensemble = ensemble
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"評估集成器 {name} 時發生錯誤: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def get_best_ensemble(self) -> Any:
        """獲取最佳集成器"""
        return self.best_ensemble
    
    def save_ensemble(self, ensemble_name: str, filepath: str):
        """
        保存集成器
        
        Args:
            ensemble_name: 集成器名稱
            filepath: 保存路徑
        """
        if ensemble_name in self.ensembles:
            joblib.dump(self.ensembles[ensemble_name], filepath)
            logger.info(f"集成器 {ensemble_name} 已保存到 {filepath}")
        else:
            raise ValueError(f"集成器 {ensemble_name} 不存在")
    
    def load_ensemble(self, ensemble_name: str, filepath: str):
        """
        載入集成器
        
        Args:
            ensemble_name: 集成器名稱
            filepath: 文件路徑
        """
        ensemble = joblib.load(filepath)
        self.add_ensemble(ensemble_name, ensemble)
        logger.info(f"集成器 {ensemble_name} 已從 {filepath} 載入")


def create_ensemble_from_config(config: Dict[str, Any], models: List[Any]) -> Any:
    """
    根據配置創建集成器
    
    Args:
        config: 集成配置
        models: 模型列表
        
    Returns:
        集成器實例
    """
    ensemble_type = config.get('type', 'voting')
    
    if ensemble_type == 'voting':
        return VotingEnsemble(
            models=models,
            voting_type=config.get('voting_type', 'soft'),
            weights=config.get('weights')
        )
    elif ensemble_type == 'stacking':
        meta_model = config.get('meta_model')
        if meta_model is None:
            meta_model = LogisticRegression(random_state=42)
        return StackingEnsemble(base_models=models, meta_model=meta_model)
    elif ensemble_type == 'weighted':
        return WeightedAverageEnsemble(
            models=models,
            weight_strategy=config.get('weight_strategy', 'accuracy')
        )
    else:
        raise ValueError(f"不支援的集成類型: {ensemble_type}")