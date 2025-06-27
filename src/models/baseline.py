"""
傳統機器學習基線模型模組

實作多種傳統機器學習算法進行情感分析，包括：
- 邏輯回歸 (Logistic Regression)
- 支持向量機 (SVM)
- 樸素貝葉斯 (Naive Bayes)
- 隨機森林 (Random Forest)

提供統一的訓練和預測接口。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

from ..utils.logger import logger


class BaselineModelManager:
    """
    傳統機器學習基線模型管理器
    
    提供多種傳統ML算法的統一訓練、預測和評估接口。
    """
    
    def __init__(self, models_dir: str = "experiments/models"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型儲存目錄
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # 定義模型配置
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_grid': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'param_grid': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf'],
                    'classifier__gamma': ['scale', 'auto']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'param_grid': {
                    'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'param_grid': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5]
                }
            }
        }
        
        # 儲存訓練好的模型
        self.trained_models = {}
        self.vectorizer = None
    
    def create_pipeline(self, model_name: str, custom_vectorizer: Optional[TfidfVectorizer] = None) -> Pipeline:
        """
        創建機器學習pipeline
        
        Args:
            model_name: 模型名稱
            custom_vectorizer: 自定義向量化器
            
        Returns:
            sklearn Pipeline對象
        """
        if model_name not in self.model_configs:
            raise ValueError(f"不支援的模型: {model_name}")
        
        # 預設TF-IDF向量化器
        if custom_vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )
        else:
            vectorizer = custom_vectorizer
        
        # 創建pipeline
        model = self.model_configs[model_name]['model']
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        return pipeline
    
    def train_model(
        self, 
        model_name: str, 
        X_train: List[str], 
        y_train: List[int],
        use_grid_search: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        訓練單個模型
        
        Args:
            model_name: 模型名稱
            X_train: 訓練文本
            y_train: 訓練標籤
            use_grid_search: 是否使用網格搜索
            cv_folds: 交叉驗證折數
            
        Returns:
            訓練結果字典
        """
        logger.info(f"開始訓練 {model_name} 模型...")
        
        # 創建pipeline
        pipeline = self.create_pipeline(model_name)
        
        if use_grid_search:
            # 網格搜索最佳參數
            param_grid = self.model_configs[model_name]['param_grid']
            
            # 添加vectorizer參數到網格搜索
            vectorizer_params = {
                'vectorizer__max_features': [5000, 10000],
                'vectorizer__ngram_range': [(1, 1), (1, 2)]
            }
            param_grid.update(vectorizer_params)
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"{model_name} 最佳參數: {best_params}")
            logger.info(f"{model_name} 最佳CV分數: {best_score:.4f}")
            
        else:
            # 直接訓練
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            
            # 計算交叉驗證分數
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds)
            best_score = cv_scores.mean()
        
        # 儲存模型
        self.trained_models[model_name] = best_model
        
        # 儲存向量化器（第一次訓練時）
        if self.vectorizer is None:
            self.vectorizer = best_model.named_steps['vectorizer']
        
        # 儲存模型到檔案
        model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        joblib.dump(best_model, model_path)
        logger.info(f"模型已儲存至: {model_path}")
        
        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': best_score,
            'model_path': model_path
        }
    
    def train_all_models(
        self, 
        X_train: List[str], 
        y_train: List[int],
        use_grid_search: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        訓練所有基線模型
        
        Args:
            X_train: 訓練文本
            y_train: 訓練標籤
            use_grid_search: 是否使用網格搜索
            
        Returns:
            所有模型的訓練結果
        """
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                result = self.train_model(
                    model_name, X_train, y_train, use_grid_search
                )
                results[model_name] = result
                logger.info(f"{model_name} 訓練完成")
                
            except Exception as e:
                logger.error(f"{model_name} 訓練失敗: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict(self, model_name: str, X_test: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用訓練好的模型進行預測
        
        Args:
            model_name: 模型名稱
            X_test: 測試文本
            
        Returns:
            (預測標籤, 預測機率)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 尚未訓練")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        
        # 獲取預測機率
        if hasattr(model.named_steps['classifier'], 'predict_proba'):
            probabilities = model.predict_proba(X_test)
        else:
            # 某些模型沒有predict_proba方法
            probabilities = None
        
        return predictions, probabilities
    
    def evaluate_model(
        self, 
        model_name: str, 
        X_test: List[str], 
        y_test: List[int]
    ) -> Dict[str, Any]:
        """
        評估模型性能
        
        Args:
            model_name: 模型名稱
            X_test: 測試文本
            y_test: 測試標籤
            
        Returns:
            評估結果字典
        """
        predictions, probabilities = self.predict(model_name, X_test)
        
        # 計算評估指標
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }
        
        if probabilities is not None:
            results['probabilities'] = probabilities
        
        logger.info(f"{model_name} 測試準確率: {accuracy:.4f}")
        
        return results
    
    def evaluate_all_models(
        self, 
        X_test: List[str], 
        y_test: List[int]
    ) -> Dict[str, Dict[str, Any]]:
        """
        評估所有訓練好的模型
        
        Args:
            X_test: 測試文本
            y_test: 測試標籤
            
        Returns:
            所有模型的評估結果
        """
        results = {}
        
        for model_name in self.trained_models.keys():
            try:
                result = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"評估 {model_name} 失敗: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def load_model(self, model_name: str, model_path: str):
        """
        從檔案載入模型
        
        Args:
            model_name: 模型名稱
            model_path: 模型檔案路徑
        """
        try:
            model = joblib.load(model_path)
            self.trained_models[model_name] = model
            logger.info(f"模型 {model_name} 載入成功")
            
        except Exception as e:
            logger.error(f"載入模型 {model_name} 失敗: {e}")
            raise
    
    def get_feature_importance(self, model_name: str, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        獲取特徵重要性（適用於支援的模型）
        
        Args:
            model_name: 模型名稱
            feature_names: 特徵名稱列表
            
        Returns:
            特徵重要性字典
        """
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 尚未訓練")
        
        model = self.trained_models[model_name]
        classifier = model.named_steps['classifier']
        
        # 獲取特徵重要性
        if hasattr(classifier, 'coef_'):
            # 線性模型（邏輯回歸、SVM）
            importance = np.abs(classifier.coef_[0])
        elif hasattr(classifier, 'feature_importances_'):
            # 樹型模型（隨機森林）
            importance = classifier.feature_importances_
        else:
            raise ValueError(f"模型 {model_name} 不支援特徵重要性分析")
        
        # 獲取特徵名稱
        if feature_names is None:
            vectorizer = model.named_steps['vectorizer']
            feature_names = vectorizer.get_feature_names_out()
        
        # 創建重要性字典
        feature_importance = dict(zip(feature_names, importance))
        
        # 按重要性排序
        sorted_features = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_features