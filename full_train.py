#!/usr/bin/env python3
"""
完整IMDB情感分析模型訓練腳本
使用全部資料集進行訓練，記錄詳細的訓練結果和時間
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

def full_train():
    """使用完整資料集訓練所有模型"""
    
    start_total_time = time.time()
    
    log_message("=== 開始完整IMDB情感分析模型訓練 ===")
    log_message("載入預處理資料...")
    
    # 載入預處理資料
    with open('data/processed/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_texts = data['train_texts']
    train_labels = data['train_labels']
    test_texts = data['test_texts']
    test_labels = data['test_labels']
    
    log_message(f"完整資料集大小:")
    log_message(f"  訓練集: {len(train_texts):,} 筆")
    log_message(f"  測試集: {len(test_texts):,} 筆")
    log_message(f"  正面樣本: {sum(train_labels):,} 筆")
    log_message(f"  負面樣本: {len(train_labels) - sum(train_labels):,} 筆")
    
    # TF-IDF向量化
    log_message("開始TF-IDF向量化...")
    vectorizer_start = time.time()
    
    vectorizer = TfidfVectorizer(
        max_features=20000,    # 使用更多特徵
        ngram_range=(1, 2),    # 1-gram和2-gram
        min_df=5,              # 最少出現5次
        max_df=0.9,            # 最多90%文檔包含
        sublinear_tf=True      # 使用對數tf
    )
    
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    
    vectorizer_time = time.time() - vectorizer_start
    
    log_message(f"TF-IDF向量化完成 (耗時: {vectorizer_time:.1f}秒)")
    log_message(f"特徵向量維度: {train_vectors.shape[1]:,}")
    log_message(f"向量稀疏度: {(1 - train_vectors.nnz / train_vectors.size) * 100:.1f}%")
    
    # 定義模型
    models = {
        'logistic_regression': {
            'model': LogisticRegression(
                random_state=42, 
                max_iter=2000,
                C=1.0,
                solver='liblinear'
            ),
            'name': '邏輯回歸'
        },
        'naive_bayes': {
            'model': MultinomialNB(alpha=1.0),
            'name': '樸素貝葉斯'
        },
        'svm_linear': {
            'model': SVC(
                kernel='linear', 
                random_state=42, 
                probability=True,
                C=1.0
            ),
            'name': 'SVM(線性核)'
        },
        'svm_rbf': {
            'model': SVC(
                kernel='rbf', 
                random_state=42, 
                probability=True,
                C=1.0,
                gamma='scale'
            ),
            'name': 'SVM(RBF核)'
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=None,
                min_samples_split=5
            ),
            'name': '隨機森林'
        }
    }
    
    results = {}
    
    log_message("\n=== 開始訓練模型 ===")
    
    for model_key, model_info in models.items():
        model = model_info['model']
        model_name = model_info['name']
        
        log_message(f"\n訓練 {model_name} ({model_key})...")
        
        # 訓練
        train_start = time.time()
        model.fit(train_vectors, train_labels)
        training_time = time.time() - train_start
        
        log_message(f"  訓練完成，耗時: {training_time:.1f}秒")
        
        # 預測
        predict_start = time.time()
        train_predictions = model.predict(train_vectors)
        test_predictions = model.predict(test_vectors)
        prediction_time = time.time() - predict_start
        
        # 計算訓練集指標
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_f1 = f1_score(train_labels, train_predictions)
        
        # 計算測試集指標
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_f1 = f1_score(test_labels, test_predictions)
        
        # AUC計算
        try:
            train_probabilities = model.predict_proba(train_vectors)[:, 1]
            test_probabilities = model.predict_proba(test_vectors)[:, 1]
            train_auc = roc_auc_score(train_labels, train_probabilities)
            test_auc = roc_auc_score(test_labels, test_probabilities)
        except Exception as e:
            log_message(f"  警告: AUC計算失敗 - {e}")
            train_auc = test_auc = 0.0
        
        # 混淆矩陣
        conf_matrix = confusion_matrix(test_labels, test_predictions)
        
        # 分類報告
        class_report = classification_report(
            test_labels, 
            test_predictions, 
            target_names=['負面', '正面'],
            output_dict=True
        )
        
        results[model_key] = {
            'model_name': model_name,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'train_auc': train_auc,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'model_params': model.get_params()
        }
        
        log_message(f"  訓練集準確率: {train_accuracy:.4f}")
        log_message(f"  測試集準確率: {test_accuracy:.4f}")
        log_message(f"  測試集F1分數: {test_f1:.4f}")
        log_message(f"  測試集AUC: {test_auc:.4f}")
        log_message(f"  預測耗時: {prediction_time:.1f}秒")
    
    total_time = time.time() - start_total_time
    
    # 保存完整結果
    log_message("\n=== 保存訓練結果 ===")
    
    # 創建結果目錄
    Path('experiments/results').mkdir(parents=True, exist_ok=True)
    
    # 完整結果
    full_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'vectorizer_time': vectorizer_time,
            'dataset_info': {
                'train_samples': len(train_texts),
                'test_samples': len(test_texts),
                'feature_dim': train_vectors.shape[1],
                'positive_samples': sum(train_labels) + sum(test_labels),
                'negative_samples': len(train_labels) + len(test_labels) - sum(train_labels) - sum(test_labels)
            }
        },
        'vectorizer_params': vectorizer.get_params(),
        'model_results': results
    }
    
    # 保存完整結果
    with open('experiments/results/full_training_results.pkl', 'wb') as f:
        pickle.dump(full_results, f)
    
    # 保存可讀的JSON格式
    json_results = {
        'experiment_info': full_results['experiment_info'],
        'vectorizer_params': full_results['vectorizer_params'],
        'model_results': {}
    }
    
    for model_key, result in results.items():
        json_results['model_results'][model_key] = {
            'model_name': result['model_name'],
            'training_time': result['training_time'],
            'prediction_time': result['prediction_time'],
            'train_accuracy': result['train_accuracy'],
            'test_accuracy': result['test_accuracy'],
            'test_f1': result['test_f1'],
            'test_auc': result['test_auc'],
            'confusion_matrix': result['confusion_matrix']
        }
    
    with open('experiments/results/full_training_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    # 打印最終總結
    log_message("\n" + "="*80)
    log_message("完整訓練結果總結")
    log_message("="*80)
    log_message(f"資料集: {len(train_texts):,} 訓練 + {len(test_texts):,} 測試")
    log_message(f"特徵維度: {train_vectors.shape[1]:,}")
    log_message(f"總訓練時間: {total_time:.1f}秒")
    log_message("")
    log_message(f"{'模型':<15} {'測試準確率':<12} {'F1分數':<10} {'AUC':<8} {'訓練時間':<10}")
    log_message("-"*70)
    
    # 按測試準確率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    for model_key, result in sorted_results:
        log_message(f"{result['model_name']:<15} {result['test_accuracy']:<12.4f} "
                   f"{result['test_f1']:<10.4f} {result['test_auc']:<8.4f} "
                   f"{result['training_time']:<10.1f}s")
    
    log_message("\n最佳模型: {} (測試準確率: {:.4f})".format(
        sorted_results[0][1]['model_name'],
        sorted_results[0][1]['test_accuracy']
    ))
    
    log_message(f"\n結果已保存至:")
    log_message(f"  - experiments/results/full_training_results.pkl")
    log_message(f"  - experiments/results/full_training_results.json")
    
    return full_results

if __name__ == "__main__":
    try:
        results = full_train()
        log_message("\n=== 完整訓練成功完成！ ===")
    except Exception as e:
        log_message(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()