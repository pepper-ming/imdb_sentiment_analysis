"""
模型評估工具
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf
import os

def evaluate_model(model, X_test, y_test, model_name, save_path='results/'):
    """全面評估模型性能"""
    os.makedirs(save_path, exist_ok=True)
    
    print(f"評估模型: {model_name}")
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"測試準確率: {test_accuracy:.4f}")
    print(f"測試損失: {test_loss:.4f}")
    
    print("\n分類報告:")
    print(classification_report(y_test, y_pred, target_names=['負面', '正面']))
    
    plot_confusion_matrix(y_test, y_pred, model_name, save_path)
    plot_roc_curve(y_test, y_pred_proba, model_name, save_path)
    plot_precision_recall_curve(y_test, y_pred_proba, model_name, save_path)
    
    results = {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    return results

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['負面', '正面'],
                yticklabels=['負面', '正面'])
    plt.title(f'{model_name} - 混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, model_name, save_path):
    """繪製ROC曲線"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲線 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title(f'{model_name} - ROC曲線')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, model_name, save_path):
    """繪製精確率-召回率曲線"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR曲線 (AUC = {pr_auc:.2f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精確率 (Precision)')
    plt.title(f'{model_name} - 精確率-召回率曲線')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_pr_curve.png'))
    plt.close()

def predict_sentiment(model, tokenizer, text, max_length=500):
    """預測單個文本的情感"""
    from src.data_preprocessing import IMDBDataPreprocessor
    
    preprocessor = IMDBDataPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_length
    )
    
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "正面" if prediction > 0.5 else "負面"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probability': prediction
    }

def compare_models(models_results, save_path='results/'):
    """比較多個模型的性能"""
    os.makedirs(save_path, exist_ok=True)
    
    model_names = list(models_results.keys())
    accuracies = [results['accuracy'] for results in models_results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.title('模型準確率比較')
    plt.xlabel('模型類型')
    plt.ylabel('準確率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'models_comparison.png'))
    plt.close()
    
    print("模型性能比較:")
    for name, results in models_results.items():
        print(f"{name}: {results['accuracy']:.4f}")

def analyze_predictions(model, X_test, y_test, tokenizer, num_samples=10):
    """分析預測結果"""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    correct_predictions = (y_pred.flatten() == y_test)
    incorrect_predictions = ~correct_predictions
    
    print(f"正確預測數量: {np.sum(correct_predictions)}")
    print(f"錯誤預測數量: {np.sum(incorrect_predictions)}")
    
    if np.sum(incorrect_predictions) > 0:
        print(f"\n錯誤預測樣本 (前{num_samples}個):")
        incorrect_indices = np.where(incorrect_predictions)[0][:num_samples]
        
        word_index = tokenizer.word_index
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        for i, idx in enumerate(incorrect_indices):
            original_text = ' '.join([reverse_word_index.get(word_id-3, '?') 
                                    for word_id in X_test[idx] if word_id > 0])
            
            print(f"\n樣本 {i+1}:")
            print(f"原文: {original_text[:200]}...")
            print(f"真實標籤: {'正面' if y_test[idx] == 1 else '負面'}")
            print(f"預測標籤: {'正面' if y_pred[idx] == 1 else '負面'}")
            print(f"預測概率: {y_pred_proba[idx][0]:.4f}")
    
    return {
        'correct_count': np.sum(correct_predictions),
        'incorrect_count': np.sum(incorrect_predictions),
        'accuracy': np.mean(correct_predictions)
    }