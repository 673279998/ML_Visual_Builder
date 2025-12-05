"""
DBSCAN聚类算法
Density-Based Spatial Clustering
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import DBSCAN as SKDBSCAN
from ..base_algorithm import BaseAlgorithm


class DBSCANClustering(BaseAlgorithm):
    """DBSCAN密度聚类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "dbscan"
        self.algorithm_type = "clustering"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'eps',
                'type': 'float',
                'default': 0.5,
                'range': [0.1, 10.0],
                'description': '邻域半径'
            },
            {
                'name': 'min_samples',
                'type': 'int',
                'default': 5,
                'range': [1, 50],
                'description': '核心点的最小邻居数'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'euclidean',
                'options': ['euclidean', 'manhattan', 'cosine'],
                'description': '距离度量'
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
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'DBSCANClustering':
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
        self.model = SKDBSCAN(**validated_params)
        self.model.fit(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        DBSCAN不支持预测新样本,返回训练时的标签
        
        Args:
            X: 测试特征
            
        Returns:
            聚类标签(-1表示噪声点)
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        # DBSCAN需要重新拟合
        return self.model.fit_predict(X)
