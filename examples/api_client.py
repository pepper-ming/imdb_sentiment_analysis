"""
API客戶端範例

演示如何使用IMDB情感分析API服務進行情感預測。
包含單個預測、批次預測、錯誤處理等功能。

使用方式:
    1. 啟動API服務: python app.py
    2. 執行客戶端: python examples/api_client.py
"""

import requests
import json
import time
from typing import List, Dict, Any

class SentimentAPIClient:
    """情感分析API客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化API客戶端
        
        Args:
            base_url: API服務基礎URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # 設置通用請求頭
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'IMDB-Sentiment-Client/1.0'
        })
    
    def check_health(self) -> Dict[str, Any]:
        """檢查API服務健康狀態"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {
                'status': 'healthy',
                'data': response.json()
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'error',
                'message': '無法連接到API服務'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_single(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """
        單個文本情感預測
        
        Args:
            text: 要分析的文本
            model_name: 指定使用的模型名稱（可選）
            
        Returns:
            預測結果字典
        """
        data = {"text": text}
        if model_name:
            data["model_name"] = model_name
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/predict", json=data)
            response_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            result['api_response_time'] = response_time
            
            return {
                'status': 'success',
                'data': result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f'請求失敗: {e}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'未知錯誤: {e}'
            }
    
    def predict_batch(self, texts: List[str], model_name: str = None) -> Dict[str, Any]:
        """
        批次文本情感預測
        
        Args:
            texts: 要分析的文本列表
            model_name: 指定使用的模型名稱（可選）
            
        Returns:
            批次預測結果字典
        """
        data = {"texts": texts}
        if model_name:
            data["model_name"] = model_name
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/predict/batch", json=data)
            response_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            result['api_response_time'] = response_time
            
            return {
                'status': 'success',
                'data': result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f'請求失敗: {e}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'未知錯誤: {e}'
            }
    
    def get_models(self) -> Dict[str, Any]:
        """獲取可用模型列表"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            
            return {
                'status': 'success',
                'data': response.json()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """載入指定模型"""
        try:
            response = self.session.post(f"{self.base_url}/models/{model_name}/load")
            response.raise_for_status()
            
            return {
                'status': 'success',
                'data': response.json()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

def demo_basic_usage():
    """基本使用演示"""
    print("🎬 IMDB情感分析API客戶端演示")
    print("=" * 50)
    
    # 創建客戶端
    client = SentimentAPIClient()
    
    # 1. 健康檢查
    print("\n🔍 1. API健康檢查...")
    health = client.check_health()
    
    if health['status'] == 'healthy':
        print("✅ API服務運行正常")
        health_data = health['data']
        print(f"   版本: {health_data.get('version', 'unknown')}")
        print(f"   運行時間: {health_data.get('uptime', 0):.2f}秒")
        print(f"   可用模型: {health_data.get('available_models', [])}")
    else:
        print(f"❌ API服務異常: {health['message']}")
        print("請確保API服務已啟動: python app.py")
        return
    
    # 2. 單個預測演示
    print("\n🎯 2. 單個預測演示...")
    
    test_cases = [
        "This movie was absolutely fantastic! Great acting and amazing plot.",
        "Terrible movie, waste of time. Poor acting and boring story.",
        "The film was okay, nothing special but not bad either."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n   測試 {i}:")
        print(f"   文本: {text}")
        
        result = client.predict_single(text)
        
        if result['status'] == 'success':
            data = result['data']
            sentiment_emoji = "😊" if data['sentiment'] == 'positive' else "😞"
            
            print(f"   預測: {sentiment_emoji} {data['sentiment']}")
            print(f"   信心度: {data['confidence']:.3f}")
            print(f"   推理時間: {data['inference_time']*1000:.1f}ms")
            print(f"   API響應時間: {data['api_response_time']*1000:.1f}ms")
        else:
            print(f"   ❌ 預測失敗: {result['message']}")
    
    # 3. 批次預測演示
    print(f"\n📊 3. 批次預測演示...")
    
    batch_texts = [
        "Amazing cinematography and stellar performances!",
        "Worst film I've ever seen in my life.",
        "It's an average movie, not great but not terrible.",
        "Absolutely loved every minute of it!",
        "Boring and predictable storyline."
    ]
    
    print(f"   批次大小: {len(batch_texts)} 個文本")
    
    batch_result = client.predict_batch(batch_texts)
    
    if batch_result['status'] == 'success':
        data = batch_result['data']
        
        print(f"   總時間: {data['total_time']*1000:.1f}ms")
        print(f"   平均時間: {data['average_time']*1000:.1f}ms/個")
        print(f"   API響應時間: {data['api_response_time']*1000:.1f}ms")
        
        print(f"\n   詳細結果:")
        for i, result in enumerate(data['results'], 1):
            sentiment_emoji = "😊" if result['sentiment'] == 'positive' else "😞"
            print(f"   {i}. {sentiment_emoji} {result['sentiment']} ({result['confidence']:.3f}) - {result['text'][:40]}...")
    else:
        print(f"   ❌ 批次預測失敗: {batch_result['message']}")
    
    # 4. 模型管理演示
    print(f"\n🤖 4. 模型管理演示...")
    
    models_result = client.get_models()
    
    if models_result['status'] == 'success':
        models = models_result['data']
        
        print(f"   可用模型: {len(models)} 個")
        for model in models:
            status = "✅ 已載入" if model['is_loaded'] else "⏸️ 未載入"
            print(f"   - {model['name']} ({model['type']}) {status}")
    else:
        print(f"   ❌ 獲取模型列表失敗: {models_result['message']}")

def demo_performance_test():
    """性能測試演示"""
    print("\n🚀 5. 性能測試演示...")
    
    client = SentimentAPIClient()
    test_text = "This is a great movie with excellent acting!"
    num_requests = 10
    
    print(f"   執行 {num_requests} 次請求進行性能測試...")
    
    times = []
    success_count = 0
    
    for i in range(num_requests):
        result = client.predict_single(test_text)
        
        if result['status'] == 'success':
            times.append(result['data']['api_response_time'] * 1000)  # 轉換為毫秒
            success_count += 1
        
        if (i + 1) % 5 == 0:
            print(f"   已完成 {i + 1}/{num_requests} 次請求")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n   📈 性能測試結果:")
        print(f"   成功率: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
        print(f"   平均響應時間: {avg_time:.1f}ms")
        print(f"   最快響應時間: {min_time:.1f}ms")
        print(f"   最慢響應時間: {max_time:.1f}ms")
        print(f"   吞吐量: {1000/avg_time:.1f} 請求/秒")
    else:
        print("   ❌ 性能測試失敗")

def interactive_mode():
    """互動模式"""
    print("\n🎮 6. 互動模式")
    print("輸入 'quit' 結束互動模式\n")
    
    client = SentimentAPIClient()
    
    while True:
        try:
            text = input("請輸入電影評論: ").strip()
            
            if text.lower() == 'quit':
                print("退出互動模式")
                break
            
            if not text:
                print("請輸入有效的文本")
                continue
            
            result = client.predict_single(text)
            
            if result['status'] == 'success':
                data = result['data']
                sentiment_emoji = "😊" if data['sentiment'] == 'positive' else "😞"
                
                print(f"預測結果: {sentiment_emoji} {data['sentiment']}")
                print(f"信心度: {data['confidence']:.3f}")
                print(f"推理時間: {data['inference_time']*1000:.1f}ms")
                
                if data.get('probabilities'):
                    probs = data['probabilities']
                    print(f"機率分佈: 負面={probs['negative']:.3f}, 正面={probs['positive']:.3f}")
                
                print("-" * 50)
            else:
                print(f"❌ 預測失敗: {result['message']}")
                
        except KeyboardInterrupt:
            print("\n退出互動模式")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

def main():
    """主函數"""
    try:
        # 基本使用演示
        demo_basic_usage()
        
        # 性能測試演示
        demo_performance_test()
        
        # 互動模式
        interactive_mode()
        
        print("\n🎉 API客戶端演示完成！")
        print("\n📚 更多信息:")
        print("- API文檔: http://localhost:8000/docs")
        print("- Web介面: http://localhost:8000/")
        print("- 健康檢查: http://localhost:8000/health")
        
    except Exception as e:
        print(f"\n❌ 演示過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()