"""
决策树分类算法
"""
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from backend.algorithms.base_algorithm import BaseAlgorithm
from typing import Dict, List, Any
import numpy as np


class DecisionTreeClassifier(BaseAlgorithm):
    """决策树分类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "decision_tree_classifier"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'criterion', 'type': 'categorical', 'default': 'gini', 
             'options': ['gini', 'entropy', 'log_loss'], 'description': '分裂质量评估标准'},
            {'name': 'max_depth', 'type': 'integer', 'default': None, 
             'range': [1, 100], 'nullable': True, 'description': '树的最大深度'},
            {'name': 'min_samples_split', 'type': 'integer', 'default': 2, 
             'range': [2, 20], 'description': '分裂内部节点所需最小样本数'},
            {'name': 'min_samples_leaf', 'type': 'integer', 'default': 1, 
             'range': [1, 20], 'description': '叶子节点所需最小样本数'},
            {'name': 'random_state', 'type': 'integer', 'default': 42, 
             'range': [0, 9999], 'description': '随机种子'}
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'DecisionTreeClassifier':
        validated_params = self.validate_hyperparameters(params)
        self.model = SKDecisionTreeClassifier(**validated_params)
        self.model.fit(X, y)
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKDecisionTreeClassifier()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)
