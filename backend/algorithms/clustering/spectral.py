"""
谱聚类算法
Spectral Clustering
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import SpectralClustering as SKSpectralClustering
from ..base_algorithm import BaseAlgorithm


class SpectralClustering(BaseAlgorithm):
    """谱聚类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "spectral"
        self.algorithm_type = "clustering"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_clusters',
                'type': 'int',
                'default': 3,
                'range': [2, 20],
                'description': '聚类数量'
            },
            {
                'name': 'affinity',
                'type': 'select',
                'default': 'rbf',
                'options': ['rbf', 'nearest_neighbors'],
                'description': '亲和度矩阵构建方法'
            },
            {
                'name': 'n_neighbors',
                'type': 'int',
                'default': 10,
                'range': [2, 50],
                'description': '最近邻数量(仅对nearest_neighbors有效)'
            },
            {
                'name': 'gamma',
                'type': 'float',
                'default': 1.0,
                'range': [0.1, 10.0],
                'description': 'RBF核的系数(仅对rbf有效)'
            },
            {
                'name': 'assign_labels',
                'type': 'select',
                'default': 'kmeans',
                'options': ['kmeans', 'discretize'],
                'description': '分配标签的策略'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'SpectralClustering':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 标签(聚类算法不需要)
            **params: 超参数
            
        Returns:
            self
        """
        # 验证超参数
        validated_params = self.validate_hyperparameters(params)
        
        # 创建并训练模型
        self.model = SKSpectralClustering(**validated_params)
        self.model.fit(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        谱聚类不支持预测新样本,返回训练时的标签
        
        Args:
            X: 测试特征
            
        Returns:
            聚类标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        # 谱聚类需要重新拟合
        return self.model.fit_predict(X)
