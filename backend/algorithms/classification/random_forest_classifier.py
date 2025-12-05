"""
随机森林分类算法
"""
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from backend.algorithms.base_algorithm import BaseAlgorithm
from typing import Dict, List, Any
import numpy as np


class RandomForestClassifier(BaseAlgorithm):
    """随机森林分类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "random_forest_classifier"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_estimators',
                'type': 'integer',
                'default': 100,
                'range': [10, 1000],
                'description': '森林中树的数量'
            },
            {
                'name': 'criterion',
                'type': 'categorical',
                'default': 'gini',
                'options': ['gini', 'entropy', 'log_loss'],
                'description': '分裂质量的评估标准'
            },
            {
                'name': 'max_depth',
                'type': 'integer',
                'default': None,
                'range': [1, 100],
                'nullable': True,
                'description': '树的最大深度'
            },
            {
                'name': 'min_samples_split',
                'type': 'integer',
                'default': 2,
                'range': [2, 20],
                'description': '分裂内部节点所需的最小样本数'
            },
            {
                'name': 'min_samples_leaf',
                'type': 'integer',
                'default': 1,
                'range': [1, 20],
                'description': '叶子节点所需的最小样本数'
            },
            {
                'name': 'max_features',
                'type': 'categorical',
                'default': 'sqrt',
                'options': ['sqrt', 'log2', None],
                'description': '寻找最佳分裂时考虑的特征数量'
            },
            {
                'name': 'bootstrap',
                'type': 'boolean',
                'default': True,
                'description': '是否使用bootstrap采样'
            },
            {
                'name': 'oob_score',
                'type': 'boolean',
                'default': False,
                'description': '是否使用袋外样本估计泛化精度'
            },
            {
                'name': 'random_state',
                'type': 'integer',
                'default': 42,
                'range': [0, 9999],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'RandomForestClassifier':
        """训练模型"""
        validated_params = self.validate_hyperparameters(params)
        self.model = SKRandomForestClassifier(**validated_params)
        self.model.fit(X, y)
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKRandomForestClassifier()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """执行预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)
