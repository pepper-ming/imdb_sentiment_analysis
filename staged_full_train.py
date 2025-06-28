#!/usr/bin/env python3
"""
分階段完整IMDB情感分析模型訓練腳本
使用全部資料集，分階段執行以便監控進度
"""

import os
import sys
import time
import pickle
import warnings
import json
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# 添加src到路徑
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def log_message(message):
    """記錄帶時間戳的訊息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    
def load_data():
    """載入和準備資料"""
    log_message("載入預處理資料...")
    
    with open('data/processed/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_texts = data['train_texts']
    train_labels = data['train_labels']
    test_texts = data['test_texts']
    test_labels = data['test_labels']
    
    log_message(f"資料載入完成:")
    log_message(f"  訓練集: {len(train_texts):,} 筆")
    log_message(f"  測試集: {len(test_texts):,} 筆")
    
    return train_texts, train_labels, test_texts, test_labels

def create_tfidf_vectors(train_texts, test_texts):
    """創建TF-IDF向量"""
    log_message("開始TF-IDF向量化...")
    start_time = time.time()
    
    vectorizer = TfidfVectorizer(
        max_features=15000,    
        ngram_range=(1, 2),    
        min_df=3,              
        max_df=0.9,            
        sublinear_tf=True      
    )
    
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    
    vectorizer_time = time.time() - start_time
    
    log_message(f"TF-IDF向量化完成:")
    log_message(f"  耗時: {vectorizer_time:.1f}秒")
    log_message(f"  特徵維度: {train_vectors.shape[1]:,}")
    log_message(f"  訓練向量形狀: {train_vectors.shape}")
    log_message(f"  測試向量形狀: {test_vectors.shape}")
    
    return train_vectors, test_vectors, vectorizer, vectorizer_time

def train_single_model(model, model_name, train_vectors, train_labels, test_vectors, test_labels):
    """訓練單個模型並返回結果"""
    log_message(f"\n開始訓練 {model_name}...")
    
    # 訓練
    train_start = time.time()
    model.fit(train_vectors, train_labels)
    training_time = time.time() - train_start
    
    log_message(f"  {model_name} 訓練完成，耗時: {training_time:.1f}秒")
    
    # 預測
    predict_start = time.time()
    train_predictions = model.predict(train_vectors)
    test_predictions = model.predict(test_vectors)
    prediction_time = time.time() - predict_start
    
    # 計算指標
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions)
    
    # AUC計算
    try:
        test_probabilities = model.predict_proba(test_vectors)[:, 1]
        test_auc = roc_auc_score(test_labels, test_probabilities)
    except:
        test_auc = 0.0
    
    log_message(f"  {model_name} 結果:")
    log_message(f"    訓練準確率: {train_accuracy:.4f}")
    log_message(f"    測試準確率: {test_accuracy:.4f}")
    log_message(f"    測試F1分數: {test_f1:.4f}")
    log_message(f"    測試AUC: {test_auc:.4f}")
    
    return {
        'model_name': model_name,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_auc': test_auc,
    }

def main():
    """主訓練函數"""
    total_start_time = time.time()
    
    log_message("=== 開始完整IMDB情感分析模型訓練 ===")
    
    # 1. 載入資料
    train_texts, train_labels, test_texts, test_labels = load_data()
    
    # 2. 創建TF-IDF向量
    train_vectors, test_vectors, vectorizer, vectorizer_time = create_tfidf_vectors(train_texts, test_texts)
    
    # 3. 定義模型 (按訓練速度順序)
    models_to_train = [
        ('naive_bayes', MultinomialNB(alpha=1.0), '樸素貝葉斯'),
        ('logistic_regression', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'), '邏輯回歸'),
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), '隨機森林'),
        ('svm_linear', SVC(kernel='linear', random_state=42, probability=True, C=1.0), 'SVM線性核')
    ]
    
    results = {}
    
    # 4. 逐個訓練模型
    for model_key, model, model_name in models_to_train:
        try:
            result = train_single_model(model, model_name, train_vectors, train_labels, test_vectors, test_labels)
            results[model_key] = result
            
            # 即時保存結果
            intermediate_results = {
                'timestamp': datetime.now().isoformat(),
                'completed_models': list(results.keys()),
                'results': results
            }
            
            with open('experiments/results/training_progress.json', 'w', encoding='utf-8') as f:
                json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            log_message(f"  錯誤: {model_name} 訓練失敗 - {str(e)}")
            continue
    
    total_time = time.time() - total_start_time
    
    # 5. 保存完整結果
    log_message("\n=== 保存最終結果 ===")
    
    final_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'vectorizer_time': vectorizer_time,
            'dataset_info': {
                'train_samples': len(train_texts),
                'test_samples': len(test_texts),
                'feature_dim': train_vectors.shape[1]
            }
        },
        'model_results': results
    }
    
    # 保存完整結果
    with open('experiments/results/complete_training_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    with open('experiments/results/complete_training_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 6. 打印最終總結
    log_message("\n" + "="*70)
    log_message("完整訓練結果總結")
    log_message("="*70)
    log_message(f"總訓練時間: {total_time:.1f}秒")
    log_message(f"資料集大小: {len(train_texts):,} 訓練 + {len(test_texts):,} 測試")
    log_message(f"TF-IDF特徵: {train_vectors.shape[1]:,} 維")
    log_message("")
    log_message(f"{'模型':<15} {'測試準確率':<12} {'F1分數':<10} {'AUC':<8} {'訓練時間':<10}")
    log_message("-"*65)
    
    # 按準確率排序
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for model_key, result in sorted_results:
            log_message(f"{result['model_name']:<15} {result['test_accuracy']:<12.4f} "
                       f"{result['test_f1']:<10.4f} {result['test_auc']:<8.4f} "
                       f"{result['training_time']:<10.1f}s")
        
        best_model = sorted_results[0][1]
        log_message(f"\n🏆 最佳模型: {best_model['model_name']} (準確率: {best_model['test_accuracy']:.4f})")
    
    log_message(f"\n結果已保存至 experiments/results/complete_training_results.json")
    log_message("=== 完整訓練完成！ ===")
    
    return final_results

if __name__ == "__main__":
    try:
        # 確保結果目錄存在
        Path('experiments/results').mkdir(parents=True, exist_ok=True)
        
        results = main()
        
    except Exception as e:
        log_message(f"\n嚴重錯誤: {str(e)}")
        import traceback
        traceback.print_exc()