"""
配置管理模組

提供專案中所有配置參數的管理功能，包括模型參數、訓練參數、資料處理參數等。
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """資料處理配置"""
    data_path: str = "data/raw"
    processed_path: str = "data/processed"
    max_length: int = 256
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    hidden_size: int = 768
    dropout_rate: float = 0.1
    freeze_base: bool = False


@dataclass
class TrainingConfig:
    """訓練配置"""
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "experiments/models"
    

@dataclass
class ProjectConfig:
    """專案總配置"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    def __post_init__(self):
        # 確保輸出目錄存在
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.data.processed_path, exist_ok=True)