"""
FastAPI推理服務

提供RESTful API接口進行情感分析預測，支援：
- 單個文本預測
- 批次文本預測
- 模型切換
- 健康檢查
- API文檔自動生成
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import time
from datetime import datetime

from .predictor import SentimentPredictor, ModelRegistry
from ..utils.logger import logger

# Pydantic模型定義
class PredictionRequest(BaseModel):
    """單個預測請求"""
    text: str = Field(..., description="要分析的文本", min_length=1, max_length=10000)
    model_name: Optional[str] = Field(None, description="指定使用的模型名稱")

class BatchPredictionRequest(BaseModel):
    """批次預測請求"""
    texts: List[str] = Field(..., description="要分析的文本列表", min_items=1, max_items=100)
    model_name: Optional[str] = Field(None, description="指定使用的模型名稱")

class PredictionResponse(BaseModel):
    """預測回應"""
    text: str
    sentiment: str = Field(..., description="情感標籤: positive 或 negative")
    confidence: float = Field(..., description="預測信心度 (0-1)")
    probabilities: Dict[str, Optional[float]] = Field(..., description="各類別機率")
    inference_time: float = Field(..., description="推理時間 (秒)")
    model_used: str = Field(..., description="使用的模型名稱")

class BatchPredictionResponse(BaseModel):
    """批次預測回應"""
    results: List[PredictionResponse]
    total_texts: int
    total_time: float
    average_time: float

class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    type: str
    path: str
    is_loaded: bool
    parameters: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    timestamp: str
    version: str
    available_models: List[str]
    current_model: Optional[str]
    uptime: float

# 創建FastAPI應用
app = FastAPI(
    title="IMDB情感分析API",
    description="基於深度學習的電影評論情感分析服務",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域變量
model_registry = None
current_predictor = None
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """應用啟動事件"""
    global model_registry, current_predictor
    
    logger.info("啟動IMDB情感分析API服務...")
    
    # 初始化模型註冊表
    model_registry = ModelRegistry(models_dir="experiments/models")
    
    # 註冊預設模型（如果存在）
    _register_default_models()
    
    # 載入預設模型
    available_models = model_registry.get_available_models()
    if available_models:
        try:
            # 嘗試載入第一個可用模型
            default_model = available_models[0]
            current_predictor = model_registry.load_model(default_model)
            logger.info(f"載入預設模型: {default_model}")
        except Exception as e:
            logger.error(f"載入預設模型失敗: {e}")
    
    logger.info("API服務啟動完成")

def _register_default_models():
    """註冊預設模型"""
    models_dir = "experiments/models"
    
    # 註冊已知的模型
    model_configs = [
        {
            "name": "distilbert_imdb",
            "path": f"{models_dir}/distilbert_imdb",
            "type": "transformer"
        },
        {
            "name": "logistic_regression_model",
            "path": f"{models_dir}/logistic_regression_model.joblib",
            "type": "sklearn"
        },
        {
            "name": "svm_model", 
            "path": f"{models_dir}/svm_model.joblib",
            "type": "sklearn"
        }
    ]
    
    for config in model_configs:
        if os.path.exists(config["path"]):
            model_registry.register_model(
                name=config["name"],
                model_path=config["path"],
                model_type=config["type"]
            )

@app.get("/", response_class=HTMLResponse)
async def root():
    """根路徑，返回簡單的HTML介面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMDB情感分析API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .input-group { margin: 20px 0; }
            textarea { width: 100%; height: 100px; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .result { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 IMDB情感分析服務</h1>
            <p>輸入電影評論文本，AI將分析其情感傾向。</p>
            
            <div class="input-group">
                <label>評論文本:</label>
                <textarea id="textInput" placeholder="輸入您的電影評論..."></textarea>
            </div>
            
            <button onclick="predict()">分析情感</button>
            
            <div id="result" class="result" style="display:none;">
                <h3>分析結果:</h3>
                <div id="resultContent"></div>
            </div>
            
            <hr>
            <p><a href="/docs">📚 API文檔</a> | <a href="/health">🔍 系統健康</a></p>
        </div>
        
        <script>
            async function predict() {
                const text = document.getElementById('textInput').value;
                if (!text.trim()) {
                    alert('請輸入文本');
                    return;
                }
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (response.ok) {
                    const sentiment = result.sentiment === 'positive' ? '😊 正面' : '😞 負面';
                    const confidence = (result.confidence * 100).toFixed(1);
                    
                    resultContent.innerHTML = `
                        <p><strong>情感:</strong> ${sentiment}</p>
                        <p><strong>信心度:</strong> ${confidence}%</p>
                        <p><strong>推理時間:</strong> ${(result.inference_time * 1000).toFixed(1)}ms</p>
                    `;
                } else {
                    resultContent.innerHTML = `<p style="color:red;">錯誤: ${result.detail}</p>`;
                }
                
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    單個文本情感預測
    
    Args:
        request: 預測請求
        
    Returns:
        預測結果
    """
    if current_predictor is None:
        raise HTTPException(status_code=503, detail="沒有可用的模型")
    
    try:
        # 如果指定了模型，嘗試切換
        if request.model_name:
            if request.model_name in model_registry.get_available_models():
                predictor = model_registry.load_model(request.model_name)
            else:
                raise HTTPException(status_code=400, detail=f"模型 {request.model_name} 不存在")
        else:
            predictor = current_predictor
        
        # 執行預測
        result = predictor.predict_single(request.text)
        
        return PredictionResponse(
            text=result['text'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            inference_time=result['inference_time'],
            model_used=model_registry.get_current_model() or "unknown"
        )
        
    except Exception as e:
        logger.error(f"預測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    批次文本情感預測
    
    Args:
        request: 批次預測請求
        
    Returns:
        批次預測結果
    """
    if current_predictor is None:
        raise HTTPException(status_code=503, detail="沒有可用的模型")
    
    try:
        start_time = time.time()
        
        # 如果指定了模型，嘗試切換
        if request.model_name:
            if request.model_name in model_registry.get_available_models():
                predictor = model_registry.load_model(request.model_name)
            else:
                raise HTTPException(status_code=400, detail=f"模型 {request.model_name} 不存在")
        else:
            predictor = current_predictor
        
        # 執行批次預測
        results = predictor.predict_batch(request.texts)
        total_time = time.time() - start_time
        
        # 格式化結果
        prediction_responses = [
            PredictionResponse(
                text=result['text'],
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                inference_time=result['inference_time'],
                model_used=model_registry.get_current_model() or "unknown"
            )
            for result in results
        ]
        
        return BatchPredictionResponse(
            results=prediction_responses,
            total_texts=len(request.texts),
            total_time=total_time,
            average_time=total_time / len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"批次預測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"批次預測失敗: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """獲取可用模型列表"""
    if model_registry is None:
        return []
    
    models_info = []
    available_models = model_registry.get_available_models()
    current_model = model_registry.get_current_model()
    
    for model_name in available_models:
        model_info = model_registry.models[model_name]
        
        models_info.append(ModelInfo(
            name=model_name,
            type=model_info['type'],
            path=model_info['path'],
            is_loaded=model_info['predictor'] is not None,
            parameters=None  # 可以添加模型參數信息
        ))
    
    return models_info

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """載入指定模型"""
    global current_predictor
    
    if model_registry is None:
        raise HTTPException(status_code=503, detail="模型註冊表未初始化")
    
    if model_name not in model_registry.get_available_models():
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
    
    try:
        current_predictor = model_registry.load_model(model_name)
        return {"message": f"模型 {model_name} 載入成功"}
        
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        raise HTTPException(status_code=500, detail=f"載入模型失敗: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        available_models=model_registry.get_available_models() if model_registry else [],
        current_model=model_registry.get_current_model() if model_registry else None,
        uptime=uptime
    )

if __name__ == "__main__":
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )