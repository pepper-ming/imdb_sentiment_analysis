"""
模型評估模組

提供全面的模型評估功能，包括：
- 分類指標計算
- 統計顯著性測試
- 模型比較和排名
- 結果視覺化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logger import logger


class ModelEvaluator:
    """
    模型評估器
    
    提供統一的模型評估和比較功能。
    """
    
    def __init__(self):
        """初始化評估器"""
        self.evaluation_results = {}
        self.model_comparisons = {}
    
    def evaluate_classification(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        評估分類模型性能
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_prob: 預測機率
            model_name: 模型名稱
            
        Returns:
            評估結果字典
        """
        # 基礎分類指標
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 詳細分類報告
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # 混淆矩陣
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        # 如果有預測機率，計算ROC-AUC
        if y_prob is not None:
            if y_prob.ndim > 1:
                # 多類別情況，取正類機率
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob
            
            try:
                auc_score = roc_auc_score(y_true, y_prob_positive)
                results['auc_roc'] = auc_score
                
                # 計算ROC曲線點
                fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive)
                results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
                
                # 計算Precision-Recall曲線
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                    y_true, y_prob_positive
                )
                results['pr_curve'] = {
                    'precision': precision_curve, 
                    'recall': recall_curve, 
                    'thresholds': pr_thresholds
                }
                
            except ValueError as e:
                logger.warning(f"無法計算AUC分數: {e}")
        
        # 儲存結果
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} 評估完成 - 準確率: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def compare_models(
        self, 
        model_results: Dict[str, Dict[str, Any]], 
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        比較多個模型的性能
        
        Args:
            model_results: 模型評估結果字典
            metric: 比較的指標
            
        Returns:
            比較結果DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'error' in results:
                continue
                
            row = {
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0)
            }
            
            if 'auc_roc' in results:
                row['AUC-ROC'] = results['auc_roc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按指定指標排序
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric, ascending=False)
        
        return comparison_df
    
    def statistical_significance_test(
        self, 
        model1_scores: List[float], 
        model2_scores: List[float],
        test_type: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        統計顯著性測試
        
        Args:
            model1_scores: 模型1的分數列表
            model2_scores: 模型2的分數列表
            test_type: 測試類型 ('ttest' 或 'wilcoxon')
            
        Returns:
            測試結果字典
        """
        if test_type == 'ttest':
            statistic, p_value = ttest_rel(model1_scores, model2_scores)
            test_name = "Paired t-test"
        elif test_type == 'wilcoxon':
            statistic, p_value = wilcoxon(model1_scores, model2_scores)
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError("test_type must be 'ttest' or 'wilcoxon'")
        
        is_significant = p_value < 0.05
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': 0.05
        }
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray, 
        model_name: str,
        class_names: List[str] = ['Negative', 'Positive']
    ) -> plt.Figure:
        """
        繪製混淆矩陣
        
        Args:
            confusion_matrix: 混淆矩陣
            model_name: 模型名稱
            class_names: 類別名稱
            
        Returns:
            matplotlib Figure對象
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(
        self, 
        model_results: Dict[str, Dict[str, Any]]
    ) -> plt.Figure:
        """
        繪製多個模型的ROC曲線
        
        Args:
            model_results: 模型評估結果字典
            
        Returns:
            matplotlib Figure對象
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            if 'roc_curve' in results:
                roc_data = results['roc_curve']
                auc_score = results.get('auc_roc', 0)
                
                ax.plot(
                    roc_data['fpr'], 
                    roc_data['tpr'],
                    label=f'{model_name} (AUC = {auc_score:.3f})',
                    linewidth=2
                )
        
        # 繪製隨機分類線
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(
        self, 
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    ) -> plt.Figure:
        """
        繪製模型比較圖
        
        Args:
            comparison_df: 模型比較DataFrame
            metrics: 要比較的指標列表
            
        Returns:
            matplotlib Figure對象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in comparison_df.columns:
                ax = axes[i]
                
                bars = ax.bar(comparison_df['Model'], comparison_df[metric])
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.set_ylim(0, 1)
                
                # 添加數值標籤
                for bar, value in zip(bars, comparison_df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # 旋轉x軸標籤
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(
        self, 
        model_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        生成評估報告
        
        Args:
            model_results: 模型評估結果字典
            
        Returns:
            評估報告字符串
        """
        report_lines = ["# 模型評估報告\n"]
        
        # 模型比較表
        comparison_df = self.compare_models(model_results)
        report_lines.append("## 模型性能比較\n")
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("\n\n")
        
        # 各模型詳細結果
        report_lines.append("## 詳細評估結果\n")
        
        for model_name, results in model_results.items():
            if 'error' in results:
                report_lines.append(f"### {model_name}\n")
                report_lines.append(f"錯誤: {results['error']}\n\n")
                continue
            
            report_lines.append(f"### {model_name}\n")
            report_lines.append(f"- 準確率: {results['accuracy']:.4f}\n")
            report_lines.append(f"- 精確率: {results['precision']:.4f}\n")
            report_lines.append(f"- 召回率: {results['recall']:.4f}\n")
            report_lines.append(f"- F1分數: {results['f1_score']:.4f}\n")
            
            if 'auc_roc' in results:
                report_lines.append(f"- AUC-ROC: {results['auc_roc']:.4f}\n")
            
            report_lines.append("\n")
        
        return "".join(report_lines)