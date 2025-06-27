"""
模型預測器模組

提供統一的模型推理接口，支援：
- 多種模型類型（傳統ML、深度學習、Transformer）
- 單個和批次預測
- 模型載入和快取
- 預測結果格式化
"""

import torch
import numpy as np
import joblib
from typing import List, Dict, Any, Union, Tuple, Optional
import os
import time
from pathlib import Path

from ..data.preprocessing import TextPreprocessor
from ..models.transformers import TransformerClassifier
from ..utils.logger import logger


class SentimentPredictor:
    """
    情感分析預測器
    
    支援多種模型類型的統一預測接口。
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'transformer',
        preprocessor_config: Optional[Dict[str, Any]] = None,
        device: str = 'cuda'
    ):
        """
        初始化預測器
        
        Args:
            model_path: 模型路徑
            model_type: 模型類型 ('transformer', 'sklearn', 'pytorch')
            preprocessor_config: 預處理器配置
            device: 計算設備
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 設置預處理器
        if preprocessor_config is None:
            if self.model_type == 'transformer':
                preprocessor_config = {
                    'remove_html': True,
                    'remove_urls': True,
                    'lowercase': False,
                    'handle_negations': False,
                    'remove_punctuation': False
                }
            else:
                preprocessor_config = {
                    'remove_html': True,
                    'remove_urls': True,
                    'lowercase': True,
                    'handle_negations': True,
                    'remove_punctuation': False
                }
        
        self.preprocessor = TextPreprocessor(**preprocessor_config)
        
        # 載入模型
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        logger.info(f"預測器初始化完成 - 模型類型: {model_type}, 設備: {self.device}")
    
    def _load_model(self):
        """載入模型"""
        try:
            if self.model_type == 'transformer':
                self._load_transformer_model()
            elif self.model_type == 'sklearn':
                self._load_sklearn_model()
            elif self.model_type == 'pytorch':
                self._load_pytorch_model()
            else:
                raise ValueError(f"不支援的模型類型: {self.model_type}")
                
            logger.info(f"模型載入成功: {self.model_path}")
            
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
    
    def _load_transformer_model(self):
        """載入Transformer模型"""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_sklearn_model(self):
        """載入sklearn模型"""
        self.model = joblib.load(self.model_path)
    
    def _load_pytorch_model(self):
        """載入PyTorch模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        # 這裡需要根據具體模型架構來載入
        # 暫時留空，可以根據需要實作
        pass
    
    def preprocess_text(self, text: str) -> str:
        """預處理單個文本"""
        return self.preprocessor.preprocess(text)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """批次預處理文本"""
        return self.preprocessor.preprocess_batch(texts)
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        預測單個文本
        
        Args:
            text: 輸入文本
            
        Returns:
            預測結果字典
        """
        start_time = time.time()
        
        # 預處理
        processed_text = self.preprocess_text(text)
        
        # 預測
        if self.model_type == 'transformer':
            result = self._predict_transformer([processed_text])
            prediction = result['predictions'][0]
            confidence = result['confidences'][0]
            probabilities = result['probabilities'][0]
        
        elif self.model_type == 'sklearn':
            result = self._predict_sklearn([processed_text])
            prediction = result['predictions'][0]
            confidence = result['confidences'][0]
            probabilities = result['probabilities'][0] if result['probabilities'] else None
        
        else:
            raise NotImplementedError(f"模型類型 {self.model_type} 尚未實作")
        
        inference_time = time.time() - start_time
        
        return {
            'text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(probabilities[0]) if probabilities is not None else None,
                'positive': float(probabilities[1]) if probabilities is not None else None
            },
            'inference_time': inference_time
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批次預測文本
        
        Args:
            texts: 文本列表
            
        Returns:
            預測結果列表
        """
        start_time = time.time()
        
        # 批次預處理
        processed_texts = self.preprocess_batch(texts)
        
        # 批次預測
        if self.model_type == 'transformer':
            result = self._predict_transformer(processed_texts)
        elif self.model_type == 'sklearn':
            result = self._predict_sklearn(processed_texts)
        else:
            raise NotImplementedError(f"模型類型 {self.model_type} 尚未實作")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(texts)
        
        # 格式化結果
        results = []
        for i, text in enumerate(texts):
            prediction = result['predictions'][i]
            confidence = result['confidences'][i]
            probabilities = result['probabilities'][i] if result['probabilities'] else None
            
            results.append({
                'text': text,
                'processed_text': processed_texts[i],
                'prediction': int(prediction),
                'sentiment': 'positive' if prediction == 1 else 'negative',
                'confidence': float(confidence),
                'probabilities': {
                    'negative': float(probabilities[0]) if probabilities is not None else None,
                    'positive': float(probabilities[1]) if probabilities is not None else None
                },
                'inference_time': avg_time
            })
        
        return results
    
    def _predict_transformer(self, texts: List[str]) -> Dict[str, Any]:
        """Transformer模型預測"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # 移動到設備
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 計算機率和預測
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.max(probabilities, dim=-1)[0]
        
        return {
            'predictions': predictions.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }
    
    def _predict_sklearn(self, texts: List[str]) -> Dict[str, Any]:
        """sklearn模型預測"""
        predictions = self.model.predict(texts)
        
        # 嘗試獲取預測機率
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(texts)
                confidences = np.max(probabilities, axis=1)
            except:
                confidences = np.ones(len(predictions)) * 0.5  # 預設信心度
        else:
            confidences = np.ones(len(predictions)) * 0.5
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'probabilities': probabilities
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': self.device,
            'preprocessor_config': {
                'remove_html': self.preprocessor.remove_html,
                'remove_urls': self.preprocessor.remove_urls,
                'lowercase': self.preprocessor.lowercase,
                'handle_negations': self.preprocessor.handle_negations,
                'remove_punctuation': self.preprocessor.remove_punctuation
            }
        }
        
        if self.model_type == 'transformer':
            info['model_name'] = self.model.config.name_or_path if hasattr(self.model, 'config') else 'unknown'
            info['max_length'] = self.tokenizer.model_max_length if self.tokenizer else 512
        
        return info


class ModelRegistry:
    """
    模型註冊表
    
    管理多個預訓練模型，支援動態載入和切換。
    """
    
    def __init__(self, models_dir: str = 'experiments/models'):
        """
        初始化模型註冊表
        
        Args:
            models_dir: 模型目錄
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.current_model = None
        
        # 自動發現模型
        self._discover_models()
    
    def _discover_models(self):
        """自動發現可用模型"""
        if not self.models_dir.exists():
            logger.warning(f"模型目錄不存在: {self.models_dir}")
            return
        
        model_configs = []
        
        # 尋找sklearn模型
        for sklearn_file in self.models_dir.glob("*.joblib"):
            model_configs.append({
                'name': sklearn_file.stem,
                'path': str(sklearn_file),
                'type': 'sklearn'
            })
        
        # 尋找PyTorch模型
        for pt_file in self.models_dir.glob("*.pth"):
            model_configs.append({
                'name': pt_file.stem,
                'path': str(pt_file),
                'type': 'pytorch'
            })
        
        # 尋找Transformer模型目錄
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / 'config.json').exists():
                model_configs.append({
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'type': 'transformer'
                })
        
        logger.info(f"發現 {len(model_configs)} 個模型")
        for config in model_configs:
            logger.info(f"  - {config['name']} ({config['type']})")
    
    def register_model(
        self, 
        name: str, 
        model_path: str, 
        model_type: str,
        preprocessor_config: Optional[Dict[str, Any]] = None
    ):
        """
        註冊模型
        
        Args:
            name: 模型名稱
            model_path: 模型路徑
            model_type: 模型類型
            preprocessor_config: 預處理器配置
        """
        self.models[name] = {
            'path': model_path,
            'type': model_type,
            'preprocessor_config': preprocessor_config,
            'predictor': None
        }
        logger.info(f"註冊模型: {name}")
    
    def load_model(self, name: str, device: str = 'cuda') -> SentimentPredictor:
        """
        載入指定模型
        
        Args:
            name: 模型名稱
            device: 計算設備
            
        Returns:
            預測器實例
        """
        if name not in self.models:
            raise ValueError(f"模型 {name} 未註冊")
        
        model_info = self.models[name]
        
        # 如果已載入，直接返回
        if model_info['predictor'] is not None:
            return model_info['predictor']
        
        # 載入模型
        predictor = SentimentPredictor(
            model_path=model_info['path'],
            model_type=model_info['type'],
            preprocessor_config=model_info['preprocessor_config'],
            device=device
        )
        
        model_info['predictor'] = predictor
        self.current_model = name
        
        return predictor
    
    def get_available_models(self) -> List[str]:
        """獲取可用模型列表"""
        return list(self.models.keys())
    
    def get_current_model(self) -> Optional[str]:
        """獲取當前模型名稱"""
        return self.current_model