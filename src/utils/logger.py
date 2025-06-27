"""
日誌系統模組

提供統一的日誌管理功能，支援不同等級的日誌記錄和格式化輸出。
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class Logger:
    """專案日誌管理器"""
    
    def __init__(self, name: str = "imdb_sentiment", log_dir: str = "experiments/logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 避免重複添加handler
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """設置日誌處理器"""
        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 檔案處理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加處理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """記錄info等級日誌"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """記錄debug等級日誌"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """記錄warning等級日誌"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """記錄error等級日誌"""
        self.logger.error(message)


# 建立全域logger實例
logger = Logger()