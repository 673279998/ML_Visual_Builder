"""
多层感知机分类算法
Multi-layer Perceptron Classifier
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.neural_network import MLPClassifier as SKMLPClassifier
from ..base_algorithm import BaseAlgorithm


class MLPClassifier(BaseAlgorithm):
    """多层感知机分类算法(神经网络)"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "mlp_classifier"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'hidden_layer_sizes',
                'type': 'tuple',
                'default': (100,),
                'description': '隐藏层的神经元数量,例如(100,)表示1层100个神经元,(100,50)表示2层'
            },
            {
                'name': 'activation',
                'type': 'select',
                'default': 'relu',
                'options': ['relu', 'tanh', 'logistic'],
                'description': '激活函数'
            },
            {
                'name': 'solver',
                'type': 'select',
                'default': 'adam',
                'options': ['adam', 'sgd', 'lbfgs'],
                'description': '优化算法'
            },
            {
                'name': 'alpha',
                'type': 'float',
                'default': 0.0001,
                'range': [1e-6, 1.0],
                'description': 'L2正则化参数'
            },
            {
                'name': 'learning_rate',
                'type': 'select',
                'default': 'constant',
                'options': ['constant', 'invscaling', 'adaptive'],
                'description': '学习率调度'
            },
            {
                'name': 'learning_rate_init',
                'type': 'float',
                'default': 0.001,
                'range': [1e-5, 1.0],
                'description': '初始学习率'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 200,
                'range': [50, 1000],
                'description': '最大迭代次数'
            },
            {
                'name': 'early_stopping',
                'type': 'bool',
                'default': False,
                'description': '是否使用早停法'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'MLPClassifier':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            **params: 超参数
            
        Returns:
            self
        """
        # print(f"DEBUG: MLPClassifier.train called with params: {params}")
        # 验证超参数
        validated_params = self.validate_hyperparameters(params)
        # print(f"DEBUG: MLPClassifier validated_params: {validated_params}")
        
        # 创建并训练模型
        self.model = SKMLPClassifier(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKMLPClassifier()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 测试特征
            
        Returns:
            预测标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 测试特征
            
        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.predict_proba(X)
