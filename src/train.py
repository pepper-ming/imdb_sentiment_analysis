"""
IMDB情感分析模型訓練腳本
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import IMDBDataPreprocessor, load_processed_data
from src.models import SentimentModels, create_callbacks
from config.config import DATA_CONFIG, MODEL_CONFIG

def plot_training_history(history, model_name, save_path='results/'):
    """繪製訓練歷史"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_training_history.png'))
    plt.close()

def train_model(model_type='simple_lstm', use_processed_data=False):
    """訓練模型"""
    print(f"開始訓練 {model_type} 模型...")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if use_processed_data and os.path.exists('data/processed/X_train.npy'):
        print("載入已處理的數據...")
        X_train, y_train, X_test, y_test, tokenizer = load_processed_data()
    else:
        print("準備數據...")
        preprocessor = IMDBDataPreprocessor(
            max_features=DATA_CONFIG['max_features'],
            max_length=DATA_CONFIG['max_length']
        )
        X_train, y_train, X_test, y_test = preprocessor.prepare_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state']
    )
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"驗證集大小: {X_val.shape}")
    print(f"測試集大小: {X_test.shape}")
    
    model_builder = SentimentModels(
        max_features=DATA_CONFIG['max_features'],
        max_length=DATA_CONFIG['max_length'],
        embedding_dim=MODEL_CONFIG['embedding_dim']
    )
    
    model = model_builder.get_model(
        model_type=model_type,
        lstm_units=MODEL_CONFIG['lstm_units'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )
    
    print(model.summary())
    
    callbacks = create_callbacks(model_type, patience=3)
    
    print("開始訓練...")
    history = model.fit(
        X_train, y_train,
        batch_size=MODEL_CONFIG['batch_size'],
        epochs=MODEL_CONFIG['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("評估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"測試準確率: {test_accuracy:.4f}")
    
    model.save(f'models/{model_type}_final.h5')
    
    plot_training_history(history, model_type)
    
    results = {
        'model_type': model_type,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'final_val_accuracy': max(history.history['val_accuracy']),
        'final_val_loss': min(history.history['val_loss'])
    }
    
    return model, history, results

def main():
    parser = argparse.ArgumentParser(description='訓練IMDB情感分析模型')
    parser.add_argument('--model', type=str, default='simple_lstm',
                       choices=['simple_lstm', 'bidirectional_lstm', 'cnn', 
                               'cnn_lstm', 'attention_lstm'],
                       help='模型類型')
    parser.add_argument('--use_processed', action='store_true',
                       help='使用已處理的數據')
    
    args = parser.parse_args()
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model, history, results = train_model(
        model_type=args.model,
        use_processed_data=args.use_processed
    )
    
    print("\n=== 訓練完成 ===")
    print(f"模型類型: {results['model_type']}")
    print(f"最終測試準確率: {results['test_accuracy']:.4f}")
    print(f"最佳驗證準確率: {results['final_val_accuracy']:.4f}")

if __name__ == "__main__":
    main()