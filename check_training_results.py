#!/usr/bin/env python3
"""
檢查和總結完整訓練結果
"""

import json
import os
from datetime import datetime

def check_results():
    """檢查當前訓練結果"""
    
    print("=== IMDB情感分析完整訓練結果檢查 ===")
    print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 檢查進度文件
    progress_file = 'experiments/results/training_progress.json'
    complete_file = 'experiments/results/complete_training_results.json'
    
    if os.path.exists(complete_file):
        print("發現完整訓練結果文件")
        with open(complete_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    elif os.path.exists(progress_file):
        print("發現進度文件，訓練可能仍在進行中")
        with open(progress_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        print("未找到訓練結果文件")
        return
    
    if 'results' in results:
        model_results = results['results']
    else:
        model_results = results.get('model_results', {})
    
    print(f"\n已完成模型數量: {len(model_results)}")
    print(f"完成的模型: {list(model_results.keys())}")
    print()
    
    # 顯示詳細結果
    print("="*80)
    print("完整訓練結果 (使用全部49,283筆訓練資料 + 9,821筆測試資料)")
    print("="*80)
    print(f"{'模型名稱':<15} {'測試準確率':<12} {'F1分數':<10} {'AUC':<8} {'訓練時間':<12} {'過擬合'}")
    print("-"*80)
    
    # 按測試準確率排序
    sorted_models = sorted(model_results.items(), 
                          key=lambda x: x[1]['test_accuracy'], 
                          reverse=True)
    
    for model_key, result in sorted_models:
        model_name = result['model_name']
        test_acc = result['test_accuracy']
        test_f1 = result['test_f1']
        test_auc = result['test_auc']
        train_time = result['training_time']
        
        # 計算過擬合程度
        train_acc = result.get('train_accuracy', 0)
        overfitting = train_acc - test_acc
        overfitting_status = "高" if overfitting > 0.15 else "中" if overfitting > 0.05 else "低"
        
        print(f"{model_name:<15} {test_acc:<12.4f} {test_f1:<10.4f} "
              f"{test_auc:<8.4f} {train_time:<12.1f}s {overfitting_status}")
    
    # 詳細分析
    print("\n" + "="*60)
    print("詳細分析")
    print("="*60)
    
    best_model = sorted_models[0]
    print(f"最佳模型: {best_model[1]['model_name']}")
    print(f"   測試準確率: {best_model[1]['test_accuracy']:.4f}")
    print(f"   F1分數: {best_model[1]['test_f1']:.4f}")
    print(f"   AUC分數: {best_model[1]['test_auc']:.4f}")
    print(f"   訓練時間: {best_model[1]['training_time']:.1f}秒")
    
    # 性能分析
    print(f"\n性能分析:")
    for model_key, result in sorted_models:
        train_acc = result.get('train_accuracy', 0)
        test_acc = result['test_accuracy']
        overfitting = train_acc - test_acc
        
        print(f"{result['model_name']}:")
        print(f"  - 訓練準確率: {train_acc:.4f}")
        print(f"  - 測試準確率: {test_acc:.4f}")
        print(f"  - 過擬合度: {overfitting:.4f} ({'過擬合' if overfitting > 0.1 else '良好'})")
        print()
    
    # 時間效率分析
    print("時間效率:")
    for model_key, result in sorted(model_results.items(), 
                                   key=lambda x: x[1]['training_time']):
        print(f"  {result['model_name']}: {result['training_time']:.1f}秒")
    
    print("\n" + "="*60)
    print("實驗總結")
    print("="*60)
    print(f"資料集: IMDB Movie Reviews (50,000筆)")
    print(f"預處理: HTML清理、否定詞處理、詞彙化")
    print(f"特徵工程: TF-IDF (15,000維)")
    print(f"訓練集: 39,283筆")
    print(f"測試集: 9,821筆")
    print(f"已完成模型: {len(model_results)}/4")

if __name__ == "__main__":
    check_results()