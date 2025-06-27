"""
IMDB情感分析快速開始範例

本腳本演示如何快速使用IMDB情感分析專案：
1. 載入資料和預處理
2. 訓練基線模型
3. 評估模型性能
4. 進行預測

使用方式:
    python examples/quick_start.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from sklearn.model_selection import train_test_split

# 導入專案模組
from src.data import IMDBDataLoader, TextPreprocessor
from src.models import BaselineModelManager
from src.evaluation import ModelEvaluator
from src.utils.logger import logger

def main():
    """主函數"""
    print("🎬 IMDB情感分析快速開始")
    print("=" * 50)
    
    # 1. 載入資料
    print("\n📊 步驟1: 載入IMDB資料集...")
    start_time = time.time()
    
    data_loader = IMDBDataLoader(cache_dir="data/raw")
    
    try:
        train_texts, train_labels, test_texts, test_labels = data_loader.load_data()
        print(f"✅ 資料載入成功 ({time.time() - start_time:.2f}秒)")
        print(f"   訓練集: {len(train_texts):,} 筆")
        print(f"   測試集: {len(test_texts):,} 筆")
        
        # 獲取資料統計
        stats = data_loader.get_data_statistics()
        print(f"   正面評論比例: {stats['train_positive_ratio']:.1%}")
        print(f"   平均文本長度: {stats['avg_train_length']:.1f} 詞")
        
    except Exception as e:
        logger.error(f"資料載入失敗: {e}")
        return
    
    # 使用較小的資料子集進行快速演示
    if len(train_texts) > 5000:
        print("\n📝 使用資料子集進行快速演示...")
        train_texts = train_texts[:5000]
        train_labels = train_labels[:5000]
        test_texts = test_texts[:1000]
        test_labels = test_labels[:1000]
    
    # 2. 資料預處理
    print("\n🔧 步驟2: 文本預處理...")
    start_time = time.time()
    
    preprocessor = TextPreprocessor(
        remove_html=True,
        remove_urls=True,
        lowercase=True,
        handle_negations=True,
        remove_punctuation=False
    )
    
    # 創建驗證集
    train_texts_final, val_texts, train_labels_final, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )
    
    # 預處理文本
    train_texts_clean = preprocessor.preprocess_batch(train_texts_final)
    val_texts_clean = preprocessor.preprocess_batch(val_texts)
    test_texts_clean = preprocessor.preprocess_batch(test_texts)
    
    print(f"✅ 預處理完成 ({time.time() - start_time:.2f}秒)")
    print(f"   最終訓練集: {len(train_texts_clean):,} 筆")
    print(f"   驗證集: {len(val_texts_clean):,} 筆")
    print(f"   測試集: {len(test_texts_clean):,} 筆")
    
    # 3. 訓練模型
    print("\n🚀 步驟3: 訓練基線模型...")
    start_time = time.time()
    
    model_manager = BaselineModelManager(models_dir="experiments/models")
    
    # 只訓練快速的模型進行演示
    quick_models = ['logistic_regression', 'naive_bayes']
    results = {}
    
    for model_name in quick_models:
        try:
            print(f"   訓練 {model_name}...")
            result = model_manager.train_model(
                model_name, 
                train_texts_clean, 
                train_labels_final,
                use_grid_search=False  # 跳過網格搜索以節省時間
            )
            results[model_name] = result
            print(f"   ✅ {model_name} CV分數: {result['cv_score']:.4f}")
            
        except Exception as e:
            print(f"   ❌ {model_name} 訓練失敗: {e}")
    
    print(f"✅ 模型訓練完成 ({time.time() - start_time:.2f}秒)")
    
    # 4. 模型評估
    print("\n📈 步驟4: 模型評估...")
    start_time = time.time()
    
    evaluator = ModelEvaluator()
    best_model_name = None
    best_accuracy = 0
    
    for model_name in results.keys():
        try:
            # 在驗證集上評估
            val_result = model_manager.evaluate_model(model_name, val_texts_clean, val_labels)
            
            # 使用評估器進行詳細分析
            eval_result = evaluator.evaluate_classification(
                val_labels,
                val_result['predictions'],
                val_result.get('probabilities'),
                model_name
            )
            
            accuracy = eval_result['accuracy']
            print(f"   {model_name}:")
            print(f"     準確率: {accuracy:.4f}")
            print(f"     F1分數: {eval_result['f1_score']:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                
        except Exception as e:
            print(f"   ❌ {model_name} 評估失敗: {e}")
    
    print(f"✅ 模型評估完成 ({time.time() - start_time:.2f}秒)")
    print(f"🏆 最佳模型: {best_model_name} (準確率: {best_accuracy:.4f})")
    
    # 5. 最佳模型測試
    if best_model_name:
        print(f"\n🎯 步驟5: 使用{best_model_name}進行測試...")
        start_time = time.time()
        
        try:
            test_result = model_manager.evaluate_model(best_model_name, test_texts_clean, test_labels)
            test_accuracy = test_result['accuracy']
            
            print(f"✅ 測試完成 ({time.time() - start_time:.2f}秒)")
            print(f"   測試集準確率: {test_accuracy:.4f}")
            
            # 生成分類報告
            eval_result = evaluator.evaluate_classification(
                test_labels,
                test_result['predictions'],
                test_result.get('probabilities'),
                f"{best_model_name}_test"
            )
            
            # 顯示混淆矩陣信息
            cm = eval_result['confusion_matrix']
            print(f"   混淆矩陣:")
            print(f"     真負例: {cm[0][0]}, 假正例: {cm[0][1]}")
            print(f"     假負例: {cm[1][0]}, 真正例: {cm[1][1]}")
            
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
    
    # 6. 互動式預測演示
    print("\n🎮 步驟6: 互動式預測演示")
    
    if best_model_name:
        sample_texts = [
            "This movie was absolutely fantastic! Great acting and amazing plot.",
            "Terrible movie, waste of time. Poor acting and boring story.",
            "The film was okay, nothing special but not bad either.",
            "One of the best movies I've ever seen! Highly recommended!",
            "Worst movie ever. Don't waste your money on this garbage."
        ]
        
        print("   預測示例:")
        for i, text in enumerate(sample_texts, 1):
            try:
                predictions, probabilities = model_manager.predict(best_model_name, [text])
                sentiment = "正面😊" if predictions[0] == 1 else "負面😞"
                confidence = probabilities[0][predictions[0]] if probabilities is not None else 0.5
                
                print(f"   {i}. 文本: {text[:50]}...")
                print(f"      預測: {sentiment} (信心度: {confidence:.3f})")
                
            except Exception as e:
                print(f"   ❌ 預測失敗: {e}")
    
    # 7. 總結
    print("\n" + "=" * 50)
    print("🎉 快速開始演示完成！")
    print("\n📋 總結:")
    
    if results:
        print("✅ 成功訓練的模型:")
        for model_name, result in results.items():
            print(f"   - {model_name}: CV分數 {result['cv_score']:.4f}")
    
    if best_model_name:
        print(f"🏆 推薦模型: {best_model_name}")
        print(f"📊 測試準確率: {test_accuracy:.4f}")
    
    print("\n🚀 下一步:")
    print("1. 執行完整的Jupyter notebooks進行深入分析")
    print("2. 訓練Transformer模型獲得更高準確率")
    print("3. 啟動API服務: python app.py")
    print("4. 瀏覽Web介面: http://localhost:8000")
    
    print("\n📚 更多信息請參考:")
    print("- README.md: 專案概述和快速開始")
    print("- USAGE.md: 詳細使用指南")
    print("- notebooks/: Jupyter實驗筆記本")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用戶中斷執行")
    except Exception as e:
        print(f"\n\n❌ 執行過程中發生錯誤: {e}")
        logger.error(f"快速開始腳本執行失敗: {e}")
    finally:
        print("\n👋 感謝使用IMDB情感分析專案！")