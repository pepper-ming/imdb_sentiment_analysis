#!/usr/bin/env python3
"""
IMDB情感分析模型訓練腳本

執行完整的資料預處理、模型訓練和評估流程
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加src到路徑
sys.path.append('src')

from src.data.dataset import IMDBDataset
from src.data.preprocessing import TextPreprocessor
from src.models.baseline import BaselineModelManager
from src.evaluation.evaluator import ModelEvaluator

def create_directories():
    """創建必要的目錄"""
    directories = [
        'data/processed',
        'experiments/models',
        'experiments/results',
        'experiments/logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("目錄結構創建完成")

def load_and_explore_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """載入和探索資料"""
    print("\n載入和探索資料...")
    
    # 載入資料
    df = pd.read_csv('data/raw/IMDB_Dataset.csv')
    
    # 基本統計
    stats = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'avg_review_length': df['review'].str.len().mean(),
        'max_review_length': df['review'].str.len().max(),
        'min_review_length': df['review'].str.len().min()
    }
    
    print(f"   資料集大小: {stats['total_samples']:,} 筆")
    print(f"   正面評論: {stats['sentiment_distribution']['positive']:,} 筆")
    print(f"   負面評論: {stats['sentiment_distribution']['negative']:,} 筆")
    print(f"   平均評論長度: {stats['avg_review_length']:.1f} 字元")
    print(f"   最長評論: {stats['max_review_length']:,} 字元")
    print(f"   最短評論: {stats['min_review_length']} 字元")
    
    return df, stats

def preprocess_data(df: pd.DataFrame) -> Tuple[List[str], List[int], List[str], List[int], Dict[str, Any]]:
    """資料預處理"""
    print("\n進行資料預處理...")
    
    # 創建預處理器
    preprocessor = TextPreprocessor(
        remove_html=True,
        remove_urls=True,
        lowercase=True,
        handle_negations=True,
        remove_stopwords=False,  # 保留停用詞，對情感分析有用
        lemmatization=True,
        min_length=10
    )
    
    # 轉換標籤
    label_map = {'positive': 1, 'negative': 0}
    labels = [label_map[sentiment] for sentiment in df['sentiment']]
    
    # 預處理文本
    print("   正在清理和預處理文本...")
    start_time = time.time()
    
    processed_texts = []
    for i, text in enumerate(df['review']):
        if i % 5000 == 0:
            print(f"   處理進度: {i}/{len(df)} ({i/len(df)*100:.1f}%)")
        
        processed_text = preprocessor.preprocess(text)
        processed_texts.append(processed_text)
    
    preprocessing_time = time.time() - start_time
    
    # 統計預處理結果
    original_lengths = [len(text) for text in df['review']]
    processed_lengths = [len(text) for text in processed_texts]
    
    preprocessing_stats = {
        'preprocessing_time': preprocessing_time,
        'original_avg_length': np.mean(original_lengths),
        'processed_avg_length': np.mean(processed_lengths),
        'length_reduction': (np.mean(original_lengths) - np.mean(processed_lengths)) / np.mean(original_lengths),
        'empty_after_processing': sum(1 for text in processed_texts if len(text.strip()) == 0)
    }
    
    # 移除空文本
    valid_indices = [i for i, text in enumerate(processed_texts) if len(text.strip()) > 0]
    processed_texts = [processed_texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    # 分割資料集 (80% 訓練, 20% 測試)
    split_idx = int(0.8 * len(processed_texts))
    
    train_texts = processed_texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = processed_texts[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"   預處理完成，耗時: {preprocessing_time:.1f} 秒")
    print(f"   訓練集: {len(train_texts):,} 筆")
    print(f"   測試集: {len(test_texts):,} 筆")
    print(f"   平均長度縮減: {preprocessing_stats['length_reduction']*100:.1f}%")
    
    # 保存預處理結果
    preprocessed_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
        'preprocessor': preprocessor,
        'stats': preprocessing_stats
    }
    
    with open('data/processed/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print("   預處理資料已保存至 data/processed/preprocessed_data.pkl")
    
    return train_texts, train_labels, test_texts, test_labels, preprocessing_stats

def train_baseline_models(train_texts: List[str], train_labels: List[int], 
                         test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
    """訓練傳統機器學習模型"""
    print("\n訓練傳統機器學習模型...")
    
    # 創建模型管理器
    model_manager = BaselineModelManager()
    
    # 訓練所有模型
    print("   正在訓練所有基線模型...")
    start_time = time.time()
    
    training_results = model_manager.train_all_models(train_texts, train_labels)
    training_time = time.time() - start_time
    
    print(f"   訓練完成，總耗時: {training_time:.1f} 秒")
    
    # 評估模型
    print("   正在評估模型性能...")
    start_time = time.time()
    
    evaluation_results = model_manager.evaluate_all_models(test_texts, test_labels)
    evaluation_time = time.time() - start_time
    
    print(f"   評估完成，耗時: {evaluation_time:.1f} 秒")
    
    # 合併結果
    results = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'training_time': training_time,
        'evaluation_time': evaluation_time
    }
    
    # 保存結果
    with open('experiments/results/baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 打印結果摘要
    print("\n傳統機器學習模型結果摘要:")
    print("-" * 70)
    print(f"{'模型':<20} {'準確率':<10} {'F1分數':<10} {'AUC':<10} {'訓練時間':<10}")
    print("-" * 70)
    
    for model_name, metrics in evaluation_results.items():
        accuracy = metrics['accuracy']
        f1 = metrics['f1_score']
        auc = metrics['auc_roc']
        train_time = training_results[model_name]['training_time']
        
        print(f"{model_name:<20} {accuracy:<10.3f} {f1:<10.3f} {auc:<10.3f} {train_time:<10.1f}s")
    
    return results

def main():
    """主函數"""
    print("IMDB情感分析模型訓練開始")
    print("=" * 50)
    
    overall_start_time = time.time()
    
    try:
        # 1. 創建目錄
        create_directories()
        
        # 2. 載入和探索資料
        df, data_stats = load_and_explore_data()
        
        # 3. 資料預處理
        train_texts, train_labels, test_texts, test_labels, preprocessing_stats = preprocess_data(df)
        
        # 4. 訓練傳統機器學習模型
        baseline_results = train_baseline_models(train_texts, train_labels, test_texts, test_labels)
        
        # 5. 保存總體統計
        overall_stats = {
            'data_stats': data_stats,
            'preprocessing_stats': preprocessing_stats,
            'baseline_results': baseline_results,
            'total_time': time.time() - overall_start_time
        }
        
        with open('experiments/results/training_summary.pkl', 'wb') as f:
            pickle.dump(overall_stats, f)
        
        print(f"\n訓練完成！總耗時: {overall_stats['total_time']:.1f} 秒")
        print(f"結果已保存至 experiments/results/")
        
        return overall_stats
        
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()