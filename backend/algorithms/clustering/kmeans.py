"""
K均值聚类算法
K-Means Clustering
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import KMeans as SKKMeans
from ..base_algorithm import BaseAlgorithm


class KMeansClustering(BaseAlgorithm):
    """K均值聚类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "kmeans"
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
                'name': 'init',
                'type': 'select',
                'default': 'k-means++',
                'options': ['k-means++', 'random'],
                'description': '初始化方法'
            },
            {
                'name': 'n_init',
                'type': 'int',
                'default': 10,
                'range': [1, 50],
                'description': '运行K-Means算法的次数,选择最佳结果'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 300,
                'range': [100, 1000],
                'description': '单次运行的最大迭代次数'
            },
            {
                'name': 'tol',
                'type': 'float',
                'default': 1e-4,
                'range': [1e-6, 1e-2],
                'description': '收敛容忍度'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'KMeansClustering':
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
        self.model = SKKMeans(**validated_params)
        self.model.fit(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测聚类标签
        
        Args:
            X: 测试特征
            
        Returns:
            聚类标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.predict(X)
