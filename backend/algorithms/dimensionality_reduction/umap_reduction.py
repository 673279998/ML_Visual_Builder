"""
UMAP降维算法
Uniform Manifold Approximation and Projection
"""
import numpy as np
from typing import Dict, Any, List
from umap import UMAP as SKUMAP
from ..base_algorithm import BaseAlgorithm


class UMAPReduction(BaseAlgorithm):
    """UMAP降维算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "umap"
        self.algorithm_type = "dimensionality_reduction"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 2,
                'range': [2, 100],
                'description': '降维后的维度数量'
            },
            {
                'name': 'n_neighbors',
                'type': 'int',
                'default': 15,
                'range': [2, 200],
                'description': '局部邻域的大小'
            },
            {
                'name': 'min_dist',
                'type': 'float',
                'default': 0.1,
                'range': [0.0, 0.99],
                'description': '低维空间中点之间的最小距离'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'euclidean',
                'options': ['euclidean', 'manhattan', 'cosine', 'correlation'],
                'description': '距离度量'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'UMAPReduction':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 标签(降维算法不需要)
            **params: 超参数
            
        Returns:
            self
        """
        # 验证超参数
        validated_params = self.validate_hyperparameters(params)
        
        # 创建并训练模型
        self.model = SKUMAP(**validated_params)
        self.model.fit(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对数据进行降维转换
        
        Args:
            X: 原始特征
            
        Returns:
            降维后的特征
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.transform(X)
