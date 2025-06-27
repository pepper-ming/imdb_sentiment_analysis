"""
IMDB情感分析預測腳本
載入訓練好的模型進行情感預測
"""

import os
import sys
import pickle
import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_preprocessing import IMDBDataPreprocessor

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path='data/processed/tokenizer.pkl'):
        """初始化預測器"""
        self.model = tf.keras.models.load_model(model_path)
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.preprocessor = IMDBDataPreprocessor()
        self.max_length = 500
        
        print(f"模型載入成功: {model_path}")
    
    def predict(self, text):
        """預測單個文本的情感"""
        cleaned_text = self.preprocessor.clean_text(text)
        
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=self.max_length
        )
        
        prediction = self.model.predict(padded_sequence, verbose=0)[0][0]
        sentiment = "正面" if prediction > 0.5 else "負面"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probability': prediction
        }
    
    def predict_batch(self, texts):
        """批量預測多個文本"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def interactive_predict(self):
        """互動式預測"""
        print("=== IMDB情感分析預測器 ===")
        print("輸入電影評論，系統將預測其情感傾向")
        print("輸入 'quit' 退出程式")
        print("-" * 40)
        
        while True:
            try:
                text = input("\n請輸入評論: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("再見！")
                    break
                
                if not text:
                    print("請輸入有效的文本")
                    continue
                
                result = self.predict(text)
                
                print(f"\n結果:")
                print(f"  情感: {result['sentiment']}")
                print(f"  信心度: {result['confidence']:.2%}")
                print(f"  原始機率: {result['probability']:.4f}")
                
            except KeyboardInterrupt:
                print("\n\n程式中斷，再見！")
                break
            except Exception as e:
                print(f"預測發生錯誤: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='IMDB情感分析預測工具')
    parser.add_argument('--model', type=str, help='模型檔案路徑')
    parser.add_argument('--text', type=str, help='要預測的文本')
    parser.add_argument('--interactive', action='store_true', help='啟動互動模式')
    
    args = parser.parse_args()
    
    if not args.model:
        model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
        if not model_files:
            print("錯誤: models 目錄中找不到模型檔案")
            print("請先訓練模型或指定模型路徑")
            return
        
        model_path = f"models/{model_files[0]}"
        print(f"自動選擇模型: {model_path}")
    else:
        model_path = args.model
    
    try:
        predictor = SentimentPredictor(model_path)
        
        if args.interactive:
            predictor.interactive_predict()
        
        elif args.text:
            result = predictor.predict(args.text)
            print(f"\n預測結果:")
            print(f"文本: {result['text']}")
            print(f"情感: {result['sentiment']}")
            print(f"信心度: {result['confidence']:.2%}")
        
        else:
            print("請使用 --text 指定要預測的文本，或使用 --interactive 啟動互動模式")
            parser.print_help()
    
    except Exception as e:
        print(f"載入模型失敗: {e}")
        print("請確認模型檔案存在且 tokenizer.pkl 已建立")

if __name__ == "__main__":
    main()