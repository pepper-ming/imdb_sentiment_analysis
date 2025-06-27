"""
IMDB情感分析API應用程式入口

使用方法:
    python app.py

或使用uvicorn:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os

# 添加專案根目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference.api import app

if __name__ == "__main__":
    import uvicorn
    
    print("🎬 啟動IMDB情感分析API服務...")
    print("📚 API文檔: http://localhost:8000/docs")
    print("🌐 Web介面: http://localhost:8000/")
    print("🔍 健康檢查: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )