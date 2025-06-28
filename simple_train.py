#!/usr/bin/env python3
"""
簡化的IMDB情感分析模型訓練腳本
避免編碼問題，直接進行模型訓練
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 添加src到路徑
sys.path.append('src')

# 導入基本模組
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def train_simple_models():
    """訓練簡化的機器學習模型"""
    print("載入預處理資料...")
    
    # 載入預處理資料
    with open('data/processed/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_texts = data['train_texts']
    train_labels = data['train_labels']
    test_texts = data['test_texts']
    test_labels = data['test_labels']
    
    print(f"訓練集大小: {len(train_texts)}")
    print(f"測試集大小: {len(test_texts)}")
    
    # TF-IDF向量化
    print("進行TF-IDF向量化...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    
    print(f"特徵向量維度: {train_vectors.shape[1]}")
    
    # 定義模型
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(kernel='rbf', random_state=42, probability=True),
        'naive_bayes': MultinomialNB(),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n開始訓練模型...")
    print("-" * 60)
    
    for model_name, model in models.items():
        print(f"訓練 {model_name}...")
        
        # 訓練
        start_time = time.time()
        model.fit(train_vectors, train_labels)
        training_time = time.time() - start_time
        
        # 預測
        start_time = time.time()
        predictions = model.predict(test_vectors)
        prediction_time = time.time() - start_time
        
        # 計算指標
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        # AUC需要機率預測
        try:
            probabilities = model.predict_proba(test_vectors)[:, 1]
            auc = roc_auc_score(test_labels, probabilities)
        except:
            auc = 0.0
        
        results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
        
        print(f"  準確率: {accuracy:.3f}")
        print(f"  F1分數: {f1:.3f}")
        print(f"  AUC: {auc:.3f}")
        print(f"  訓練時間: {training_time:.1f}s")
        print()
    
    # 保存結果
    Path('experiments/results').mkdir(parents=True, exist_ok=True)
    
    with open('experiments/results/simple_training_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'vectorizer': vectorizer
        }, f)
    
    # 打印總結
    print("=" * 60)
    print("訓練結果總結")
    print("=" * 60)
    print(f"{'模型':<20} {'準確率':<10} {'F1分數':<10} {'AUC':<10} {'訓練時間':<10}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {metrics['auc_roc']:<10.3f} "
              f"{metrics['training_time']:<10.1f}s")
    
    return results

if __name__ == "__main__":
    results = train_simple_models()
    print("\n訓練完成！")