"""
Lasso回归算法
Lasso Regression
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import Lasso as SKLasso
from ..base_algorithm import BaseAlgorithm


class LassoRegression(BaseAlgorithm):
    """Lasso回归算法(L1正则化线性回归)"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "lasso_regression"
        self.algorithm_type = "regression"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'alpha',
                'type': 'float',
                'default': 1.0,
                'range': [0.001, 100.0],
                'description': 'L1正则化强度,值越大正则化越强'
            },
            {
                'name': 'fit_intercept',
                'type': 'bool',
                'default': True,
                'description': '是否拟合截距'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 1000,
                'range': [100, 10000],
                'description': '最大迭代次数'
            },
            {
                'name': 'tol',
                'type': 'float',
                'default': 1e-4,
                'range': [1e-6, 1e-2],
                'description': '优化的容忍度'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'LassoRegression':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **params: 超参数
            
        Returns:
            self
        """
        # 验证超参数
        validated_params = self.validate_hyperparameters(params)
        
        # 创建并训练模型
        self.model = SKLasso(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKLasso()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 测试特征
            
        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.predict(X)
