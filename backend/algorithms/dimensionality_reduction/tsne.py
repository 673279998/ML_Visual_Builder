"""
t-SNE降维算法
t-Distributed Stochastic Neighbor Embedding
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.manifold import TSNE as SKTSNE
from ..base_algorithm import BaseAlgorithm


class TSNEReduction(BaseAlgorithm):
    """t-SNE降维算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "tsne"
        self.algorithm_type = "dimensionality_reduction"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 2,
                'range': [2, 3],
                'description': '降维后的维度数量(通常为2或3)'
            },
            {
                'name': 'perplexity',
                'type': 'float',
                'default': 30.0,
                'range': [5.0, 50.0],
                'description': '困惑度,与最近邻数量相关'
            },
            {
                'name': 'learning_rate',
                'type': 'float',
                'default': 200.0,
                'range': [10.0, 1000.0],
                'description': '学习率'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 1000,
                'range': [250, 5000],
                'description': '优化迭代次数'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'euclidean',
                'options': ['euclidean', 'manhattan', 'cosine'],
                'description': '距离度量'
            },
            {
                'name': 'init',
                'type': 'select',
                'default': 'random',
                'options': ['random', 'pca'],
                'description': '初始化方法'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'TSNEReduction':
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
        self.model = SKTSNE(**validated_params)
        # t-SNE在fit时直接计算嵌入
        self.embedding_ = self.model.fit_transform(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        t-SNE不支持对新数据进行转换,返回训练时的嵌入
        
        Args:
            X: 原始特征
            
        Returns:
            降维后的特征
        """
        if not hasattr(self, 'embedding_'):
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        # t-SNE不支持transform,只能返回训练时的结果
        return self.embedding_
