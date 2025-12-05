"""
逻辑回归算法
"""
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from backend.algorithms.base_algorithm import BaseAlgorithm
from typing import Dict, List, Any, Optional
import numpy as np


class LogisticRegression(BaseAlgorithm):
    """逻辑回归算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "logistic_regression"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'C',
                'type': 'float',
                'default': 1.0,
                'range': [0.001, 100.0],
                'description': '正则化强度的倒数,值越小正则化越强'
            },
            {
                'name': 'penalty',
                'type': 'categorical',
                'default': 'l2',
                'options': ['l1', 'l2', 'elasticnet', 'none'],
                'description': '正则化类型'
            },
            {
                'name': 'solver',
                'type': 'categorical',
                'default': 'lbfgs',
                'options': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'description': '优化算法'
            },
            {
                'name': 'max_iter',
                'type': 'integer',
                'default': 100,
                'range': [10, 1000],
                'description': '最大迭代次数'
            },
            {
                'name': 'multi_class',
                'type': 'categorical',
                'default': 'auto',
                'options': ['auto', 'ovr', 'multinomial'],
                'description': '多分类策略'
            },
            {
                'name': 'class_weight',
                'type': 'categorical',
                'default': None,
                'options': [None, 'balanced'],
                'nullable': True,
                'description': '类别权重'
            },
            {
                'name': 'random_state',
                'type': 'integer',
                'default': 42,
                'range': [0, 9999],
                'description': '随机种子'
            }
        ]
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKLogisticRegression()

    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'LogisticRegression':
        """训练模型"""
        # 验证超参数
        validated_params = self.validate_hyperparameters(params)
        
        # 创建模型
        self.model = SKLogisticRegression(**validated_params)
        
        # 训练模型
        self.model.fit(X, y)
        
        return self
    
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
