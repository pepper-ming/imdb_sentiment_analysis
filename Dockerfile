# IMDB情感分析API服務 Docker鏡像
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 設置環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 安裝系統依賴
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        && rm -rf /var/lib/apt/lists/*

# 複製requirements文件並安裝Python依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 下載spaCy語言模型
RUN python -m spacy download en_core_web_sm

# 複製專案代碼
COPY . .

# 創建非root用戶
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# 創建必要目錄
RUN mkdir -p experiments/models experiments/logs data/raw data/processed

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD ["python", "app.py"]