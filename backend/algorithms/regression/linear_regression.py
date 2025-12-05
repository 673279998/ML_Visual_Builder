"""
线性回归算法
"""
from sklearn.linear_model import LinearRegression as SKLinearRegression
from backend.algorithms.base_algorithm import BaseAlgorithm
from typing import Dict, List, Any
import numpy as np


class LinearRegression(BaseAlgorithm):
    """线性回归算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "linear_regression"
        self.algorithm_type = "regression"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'fit_intercept',
                'type': 'boolean',
                'default': True,
                'description': '是否计算截距'
            },
            {
                'name': 'copy_X',
                'type': 'boolean',
                'default': True,
                'description': '是否复制X'
            },
            {
                'name': 'positive',
                'type': 'boolean',
                'default': False,
                'description': '是否强制系数为正'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'LinearRegression':
        """训练模型"""
        validated_params = self.validate_hyperparameters(params)
        self.model = SKLinearRegression(**validated_params)
        self.model.fit(X, y)
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKLinearRegression()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """执行预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X)
