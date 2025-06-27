"""
IMDB情感分析專案主執行腳本
提供簡單的命令行界面進行數據預處理、模型訓練和預測
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

def setup_project_path():
    """設定專案路徑"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def preprocess_data():
    """執行數據預處理"""
    setup_project_path()
    from src.data_preprocessing import IMDBDataPreprocessor
    
    print("開始數據預處理...")
    preprocessor = IMDBDataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.prepare_data()
    print("數據預處理完成！")
    return True

def train_model(model_type='simple_lstm', epochs=5):
    """訓練模型"""
    setup_project_path()
    from src.data_preprocessing import load_processed_data
    from src.models import SentimentModels, create_callbacks
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    import numpy as np
    
    print(f"開始訓練 {model_type} 模型...")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        X_train, y_train, X_test, y_test, tokenizer = load_processed_data()
    except FileNotFoundError:
        print("找不到預處理數據，正在進行數據預處理...")
        preprocess_data()
        X_train, y_train, X_test, y_test, tokenizer = load_processed_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model_builder = SentimentModels()
    model = model_builder.get_model(model_type=model_type)
    
    print(f"模型架構: {model_type}")
    print(f"訓練集大小: {X_train.shape}")
    print(f"驗證集大小: {X_val.shape}")
    
    callbacks = create_callbacks(model_type, patience=2)
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n訓練完成！測試準確率: {test_accuracy:.4f}")
    
    model.save(f'models/{model_type}_final.h5')
    print(f"模型已保存至: models/{model_type}_final.h5")

def predict_text(text, model_path=None):
    """預測單個文本的情感"""
    setup_project_path()
    import tensorflow as tf
    import pickle
    from src.evaluation import predict_sentiment
    
    if model_path is None:
        model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
        if not model_files:
            print("錯誤: 找不到訓練好的模型")
            return None
        model_path = f"models/{model_files[0]}"
    
    try:
        model = tf.keras.models.load_model(model_path)
        with open('data/processed/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        result = predict_sentiment(model, tokenizer, text)
        print(f"文本: {text}")
        print(f"預測情感: {result['sentiment']}")
        print(f"信心度: {result['confidence']:.4f}")
        return result
        
    except Exception as e:
        print(f"預測失敗: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='IMDB情感分析工具')
    parser.add_argument('action', choices=['preprocess', 'train', 'predict'], 
                       help='執行動作: preprocess(預處理), train(訓練), predict(預測)')
    parser.add_argument('--model', type=str, default='simple_lstm',
                       choices=['simple_lstm', 'bidirectional_lstm', 'cnn', 'cnn_lstm'],
                       help='模型類型')
    parser.add_argument('--epochs', type=int, default=5, help='訓練輪數')
    parser.add_argument('--text', type=str, help='要預測的文本')
    parser.add_argument('--model_path', type=str, help='模型檔案路徑')
    
    args = parser.parse_args()
    
    if args.action == 'preprocess':
        preprocess_data()
    
    elif args.action == 'train':
        train_model(args.model, args.epochs)
    
    elif args.action == 'predict':
        if not args.text:
            print("錯誤: 請使用 --text 參數提供要預測的文本")
            return
        predict_text(args.text, args.model_path)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()