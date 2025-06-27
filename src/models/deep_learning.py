"""
深度學習模型模組

實作多種深度學習架構進行情感分析，包括：
- TextCNN: 卷積神經網路文本分類
- BiLSTM: 雙向長短期記憶網路  
- GRU: 門控循環單元
- 支援預訓練詞嵌入整合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os

from ..utils.logger import logger


class TextCNN(nn.Module):
    """
    文本卷積神經網路 (TextCNN)
    
    使用多個不同大小的卷積核捕捉不同長度的n-gram特徵。
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        初始化TextCNN模型
        
        Args:
            vocab_size: 詞彙表大小
            embed_dim: 詞嵌入維度
            num_filters: 每個卷積核的數量
            filter_sizes: 卷積核大小列表
            num_classes: 分類類別數
            dropout_rate: Dropout比率
            pretrained_embeddings: 預訓練詞嵌入
        """
        super(TextCNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # 允許微調
        
        # 卷積層
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        # Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全連接層
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型權重"""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入文本序列 [batch_size, seq_len]
            
        Returns:
            分類預測結果 [batch_size, num_classes]
        """
        # 詞嵌入: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # 轉置為卷積層所需格式: [batch_size, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # 卷積和池化
        conv_outputs = []
        for conv in self.convs:
            # 卷積: [batch_size, num_filters, conv_seq_len]
            conv_out = F.relu(conv(embedded))
            # 最大池化: [batch_size, num_filters, 1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            # 壓縮維度: [batch_size, num_filters]
            pooled = pooled.squeeze(2)
            conv_outputs.append(pooled)
        
        # 連接所有卷積輸出: [batch_size, len(filter_sizes) * num_filters]
        concat_output = torch.cat(conv_outputs, dim=1)
        
        # Dropout和全連接
        output = self.dropout(concat_output)
        logits = self.fc(output)
        
        return logits


class BiLSTM(nn.Module):
    """
    雙向長短期記憶網路 (BiLSTM)
    
    使用雙向LSTM捕捉序列的前後文信息。
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        初始化BiLSTM模型
        
        Args:
            vocab_size: 詞彙表大小
            embed_dim: 詞嵌入維度
            hidden_dim: LSTM隱藏層維度
            num_layers: LSTM層數
            num_classes: 分類類別數
            dropout_rate: Dropout比率
            pretrained_embeddings: 預訓練詞嵌入
        """
        super(BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        
        # BiLSTM層
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全連接層 (雙向所以是 hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型權重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入文本序列 [batch_size, seq_len]
            
        Returns:
            分類預測結果 [batch_size, num_classes]
        """
        # 詞嵌入: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # LSTM: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最後一個時步的隱藏狀態
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # 取最後一層的前向和後向隱藏狀態
        forward_hidden = hidden[-2]  # 前向最後一層
        backward_hidden = hidden[-1]  # 後向最後一層
        
        # 連接前向和後向隱藏狀態: [batch_size, hidden_dim * 2]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout和全連接
        output = self.dropout(final_hidden)
        logits = self.fc(output)
        
        return logits


class GRUClassifier(nn.Module):
    """
    門控循環單元分類器 (GRU)
    
    使用GRU進行序列建模，計算複雜度比LSTM更低。
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        初始化GRU分類器
        
        Args:
            vocab_size: 詞彙表大小
            embed_dim: 詞嵌入維度
            hidden_dim: GRU隱藏層維度
            num_layers: GRU層數
            num_classes: 分類類別數
            dropout_rate: Dropout比率
            pretrained_embeddings: 預訓練詞嵌入
        """
        super(GRUClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        
        # GRU層
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全連接層
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型權重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入文本序列 [batch_size, seq_len]
            
        Returns:
            分類預測結果 [batch_size, num_classes]
        """
        # 詞嵌入: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # GRU: [batch_size, seq_len, hidden_dim * 2]
        gru_out, hidden = self.gru(embedded)
        
        # 使用最後一個時步的隱藏狀態
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        forward_hidden = hidden[-2]  # 前向最後一層
        backward_hidden = hidden[-1]  # 後向最後一層
        
        # 連接前向和後向隱藏狀態
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout和全連接
        output = self.dropout(final_hidden)
        logits = self.fc(output)
        
        return logits


class DeepLearningModelManager:
    """
    深度學習模型管理器
    
    提供統一的深度學習模型創建、訓練和管理接口。
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            device: 計算設備 ('cuda' 或 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用設備: {self.device}")
        
        # 模型配置
        self.model_configs = {
            'textcnn': {
                'class': TextCNN,
                'default_params': {
                    'embed_dim': 128,
                    'num_filters': 100,
                    'filter_sizes': [3, 4, 5],
                    'dropout_rate': 0.5
                }
            },
            'bilstm': {
                'class': BiLSTM,
                'default_params': {
                    'embed_dim': 128,
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'dropout_rate': 0.3
                }
            },
            'gru': {
                'class': GRUClassifier,
                'default_params': {
                    'embed_dim': 128,
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'dropout_rate': 0.3
                }
            }
        }
        
        self.models = {}
    
    def create_model(
        self, 
        model_type: str, 
        vocab_size: int, 
        num_classes: int = 2,
        **kwargs
    ) -> nn.Module:
        """
        創建深度學習模型
        
        Args:
            model_type: 模型類型 ('textcnn', 'bilstm', 'gru')
            vocab_size: 詞彙表大小
            num_classes: 分類類別數
            **kwargs: 其他模型參數
            
        Returns:
            PyTorch模型實例
        """
        if model_type not in self.model_configs:
            raise ValueError(f"不支援的模型類型: {model_type}")
        
        config = self.model_configs[model_type]
        model_class = config['class']
        default_params = config['default_params'].copy()
        
        # 更新參數
        default_params.update(kwargs)
        default_params['vocab_size'] = vocab_size
        default_params['num_classes'] = num_classes
        
        # 創建模型
        model = model_class(**default_params)
        model = model.to(self.device)
        
        logger.info(f"創建 {model_type} 模型，參數量: {self.count_parameters(model):,}")
        
        return model
    
    def count_parameters(self, model: nn.Module) -> int:
        """計算模型參數量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def load_pretrained_embeddings(
        self, 
        vocab: Dict[str, int], 
        embedding_file: str,
        embed_dim: int = 100
    ) -> torch.Tensor:
        """
        載入預訓練詞嵌入
        
        Args:
            vocab: 詞彙表字典
            embedding_file: 詞嵌入檔案路徑
            embed_dim: 嵌入維度
            
        Returns:
            預訓練嵌入矩陣
        """
        embeddings = torch.randn(len(vocab), embed_dim)
        
        if os.path.exists(embedding_file):
            logger.info(f"載入預訓練詞嵌入: {embedding_file}")
            
            with open(embedding_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == embed_dim + 1:
                        word = parts[0]
                        if word in vocab:
                            idx = vocab[word]
                            embeddings[idx] = torch.tensor([float(x) for x in parts[1:]])
        else:
            logger.warning(f"詞嵌入檔案不存在: {embedding_file}")
        
        return embeddings