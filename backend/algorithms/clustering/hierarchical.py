"""
层次聚类算法
Hierarchical Clustering (Agglomerative)
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import AgglomerativeClustering as SKAgglomerativeClustering
from ..base_algorithm import BaseAlgorithm


class HierarchicalClustering(BaseAlgorithm):
    """层次聚类算法(凝聚型)"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "hierarchical"
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
                'name': 'linkage',
                'type': 'select',
                'default': 'ward',
                'options': ['ward', 'complete', 'average', 'single'],
                'description': '链接方法'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'euclidean',
                'options': ['euclidean', 'manhattan', 'cosine'],
                'description': '距离度量(ward只能使用euclidean)'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'HierarchicalClustering':
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
        
        # ward只能使用euclidean距离
        if validated_params.get('linkage') == 'ward':
            validated_params['metric'] = 'euclidean'
        
        # 创建并训练模型
        self.model = SKAgglomerativeClustering(**validated_params)
        self.model.fit(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        层次聚类不支持预测新样本,返回训练时的标签
        
        Args:
            X: 测试特征
            
        Returns:
            聚类标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        # 层次聚类需要重新拟合
        return self.model.fit_predict(X)
