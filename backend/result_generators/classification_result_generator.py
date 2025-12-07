"""  
分类算法结果生成器
"""
import numpy as np
import os
import json
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from .base_result_generator import BaseResultGenerator
from backend.config import DATA_DIR


class ClassificationResultGenerator(BaseResultGenerator):
    """分类算法结果生成器"""
    
    def __init__(self):
        super().__init__()
        self.result_type = "classification"
    
    def generate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成分类性能指标
        
        指标包括:
        - 准确率 (accuracy)
        - 精确率 (precision)
        - 召回率 (recall)
        - F1分数 (f1_score)
        - ROC-AUC (roc_auc)
        """
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # 多分类/二分类指标
        unique_classes = len(np.unique(y_true))
        average_method = 'binary' if unique_classes == 2 else 'weighted'
        
        metrics['precision'] = float(precision_score(
            y_true, y_pred, average=average_method, zero_division=0
        ))
        metrics['recall'] = float(recall_score(
            y_true, y_pred, average=average_method, zero_division=0
        ))
        metrics['f1_score'] = float(f1_score(
            y_true, y_pred, average=average_method, zero_division=0
        ))
        
        # ROC-AUC (需要概率预测)
        roc_auc_generated = False
        if hasattr(model, 'predict_proba') and X_test is not None:
            try:
                y_prob = model.predict_proba(X_test)
                if unique_classes == 2:
                    # 二分类
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    # 多分类
                    metrics['roc_auc'] = float(roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    ))
                roc_auc_generated = True
            except Exception as e:
                metrics['roc_auc'] = None
                metrics['roc_auc_error'] = str(e)
                roc_auc_generated = True
        
        # 如果没有生成roc_auc，设置默认值
        if not roc_auc_generated:
            metrics['roc_auc'] = None
        
        # 类别统计
        metrics['num_classes'] = int(unique_classes)
        metrics['class_distribution'] = {
            int(cls): int(np.sum(y_true == cls)) 
            for cls in np.unique(y_true)
        }
        
        return metrics
    
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成分类可视化数据
        
        可视化包括:
        - 混淆矩阵 (confusion_matrix)
        - ROC曲线数据 (roc_curve)
        - PR曲线数据 (precision_recall_curve)
        - 类别预测分布 (class_prediction_distribution)
        """
        visualizations = {}
        
        # 1. 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        visualizations['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': np.unique(y_true).tolist(),
            'shape': cm.shape
        }
        
        # 2. ROC曲线和PR曲线(需要概率预测)
        if hasattr(model, 'predict_proba') and X_test is not None:
            try:
                y_prob = model.predict_proba(X_test)
                unique_classes = len(np.unique(y_true))
                
                if unique_classes == 2:
                    # 二分类ROC曲线
                    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
                    visualizations['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': float(roc_auc_score(y_true, y_prob[:, 1]))
                    }
                    
                    # PR曲线
                    precision, recall, pr_thresholds = precision_recall_curve(
                        y_true, y_prob[:, 1]
                    )
                    visualizations['pr_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': pr_thresholds.tolist()
                    }
                else:
                    # 多分类ROC曲线(One-vs-Rest)
                    roc_data = {}
                    for i, class_label in enumerate(np.unique(y_true)):
                        y_true_binary = (y_true == class_label).astype(int)
                        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                        roc_data[f'class_{int(class_label)}'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(roc_auc_score(y_true_binary, y_prob[:, i]))
                        }
                    visualizations['roc_curve_multiclass'] = roc_data
                
                # 3. 预测概率分布
                visualizations['probability_distribution'] = {
                    'probabilities': y_prob.tolist()[:100],  # 限制前100个样本
                    'predicted_classes': y_pred.tolist()[:100],
                    'true_classes': y_true.tolist()[:100]
                }
                
            except Exception as e:
                visualizations['probability_error'] = str(e)
        
        # 4. 类别预测分布
        visualizations['class_prediction_distribution'] = {
            int(cls): int(np.sum(y_pred == cls)) 
            for cls in np.unique(y_pred)
        }
        
        # 5. 预测错误分析
        misclassified_indices = np.where(y_true != y_pred)[0]
        visualizations['error_analysis'] = {
            'total_errors': int(len(misclassified_indices)),
            'error_rate': float(len(misclassified_indices) / len(y_true)),
            'error_distribution': {}
        }
        
        # 错误类型分布
        for true_class in np.unique(y_true):
            for pred_class in np.unique(y_pred):
                if true_class != pred_class:
                    count = np.sum((y_true == true_class) & (y_pred == pred_class))
                    if count > 0:
                        key = f'true_{int(true_class)}_pred_{int(pred_class)}'
                        visualizations['error_analysis']['error_distribution'][key] = int(count)
        
        return visualizations
    
    def generate_algorithm_specific_info(self, model: Any, algorithm_name: str = None) -> Dict[str, Any]:
        """
        生成算法特定的额外信息
        """
        info = {}
        
        # CatBoost特定信息
        if algorithm_name and 'catboost' in algorithm_name.lower():
            info['algorithm'] = 'CatBoost'
            
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                info['feature_importances'] = model.feature_importances_.tolist()
            
            # 获取训练信息(如果catboost_info目录存在)
            catboost_info_dir = DATA_DIR / 'catboost_info'
            if catboost_info_dir.exists():
                info['training_info'] = {
                    'info_dir': str(catboost_info_dir),
                    'has_training_data': True
                }
                
                # 尝试读取learn_error.tsv
                learn_error_file = catboost_info_dir / 'learn_error.tsv'
                if learn_error_file.exists():
                    try:
                        with open(learn_error_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                # 只读取最后几行
                                info['training_info']['final_metrics'] = lines[-1].strip()
                    except Exception as e:
                        info['training_info']['error'] = str(e)
            
            # 获取模型参数
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info['model_params'] = {
                    'iterations': params.get('iterations', 'N/A'),
                    'learning_rate': params.get('learning_rate', 'N/A'),
                    'depth': params.get('depth', 'N/A'),
                    'l2_leaf_reg': params.get('l2_leaf_reg', 'N/A')
                }
        
        # XGBoost特定信息
        elif algorithm_name and 'xgboost' in algorithm_name.lower():
            info['algorithm'] = 'XGBoost'
            if hasattr(model, 'feature_importances_'):
                info['feature_importances'] = model.feature_importances_.tolist()
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info['model_params'] = {
                    'n_estimators': params.get('n_estimators', 'N/A'),
                    'max_depth': params.get('max_depth', 'N/A'),
                    'learning_rate': params.get('learning_rate', 'N/A')
                }
        
        # LightGBM特定信息
        elif algorithm_name and 'lightgbm' in algorithm_name.lower():
            info['algorithm'] = 'LightGBM'
            if hasattr(model, 'feature_importances_'):
                info['feature_importances'] = model.feature_importances_.tolist()
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info['model_params'] = {
                    'n_estimators': params.get('n_estimators', 'N/A'),
                    'max_depth': params.get('max_depth', 'N/A'),
                    'learning_rate': params.get('learning_rate', 'N/A')
                }
        
        # 随机森林/决策树特定信息
        elif algorithm_name and ('forest' in algorithm_name.lower() or 'tree' in algorithm_name.lower()):
            if hasattr(model, 'feature_importances_'):
                info['feature_importances'] = model.feature_importances_.tolist()
            if hasattr(model, 'get_params'):
                params = model.get_params()
                if 'forest' in algorithm_name.lower():
                    info['model_params'] = {
                        'n_estimators': params.get('n_estimators', 'N/A'),
                        'max_depth': params.get('max_depth', 'N/A')
                    }
                else:
                    info['model_params'] = {
                        'max_depth': params.get('max_depth', 'N/A'),
                        'min_samples_split': params.get('min_samples_split', 'N/A')
                    }
        
        # SVM特定信息
        elif algorithm_name and 'svm' in algorithm_name.lower():
            info['algorithm'] = 'SVM'
            if hasattr(model, 'support_vectors_'):
                info['n_support_vectors'] = len(model.support_vectors_)
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info['model_params'] = {
                    'kernel': params.get('kernel', 'N/A'),
                    'C': params.get('C', 'N/A'),
                    'gamma': params.get('gamma', 'N/A')
                }
        
        # 神经网络特定信息
        elif algorithm_name and 'mlp' in algorithm_name.lower():
            info['algorithm'] = 'MLP'
            if hasattr(model, 'loss_curve_'):
                info['training_loss'] = model.loss_curve_
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info['model_params'] = {
                    'hidden_layer_sizes': params.get('hidden_layer_sizes', 'N/A'),
                    'activation': params.get('activation', 'N/A'),
                    'alpha': params.get('alpha', 'N/A')
                }
        
        return info
