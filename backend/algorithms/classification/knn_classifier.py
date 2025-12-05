"""
K近邻分类算法
K-Nearest Neighbors Classifier
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from ..base_algorithm import BaseAlgorithm


class KNNClassifier(BaseAlgorithm):
    """K近邻分类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "knn_classifier"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_neighbors',
                'type': 'int',
                'default': 5,
                'range': [1, 50],
                'description': '邻居的数量'
            },
            {
                'name': 'weights',
                'type': 'select',
                'default': 'uniform',
                'options': ['uniform', 'distance'],
                'description': '权重函数,uniform表示所有点权重相等,distance表示权重与距离成反比'
            },
            {
                'name': 'algorithm',
                'type': 'select',
                'default': 'auto',
                'options': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'description': '计算最近邻的算法'
            },
            {
                'name': 'leaf_size',
                'type': 'int',
                'default': 30,
                'range': [10, 100],
                'description': 'BallTree或KDTree的叶子大小'
            },
            {
                'name': 'p',
                'type': 'int',
                'default': 2,
                'range': [1, 3],
                'description': 'Minkowski距离的幂参数,1为曼哈顿距离,2为欧氏距离'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'minkowski',
                'options': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                'description': '距离度量'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'KNNClassifier':
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
        self.model = SKKNeighborsClassifier(**validated_params)
        self.model.fit(X, y)
        
        return self

    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKKNeighborsClassifier()
    
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
