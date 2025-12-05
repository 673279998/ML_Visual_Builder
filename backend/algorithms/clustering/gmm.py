"""
高斯混合模型聚类
Gaussian Mixture Model
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.mixture import GaussianMixture as SKGaussianMixture
from ..base_algorithm import BaseAlgorithm


class GMMClustering(BaseAlgorithm):
    """高斯混合模型聚类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "gmm"
        self.algorithm_type = "clustering"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 3,
                'range': [1, 20],
                'description': '混合成分数量(聚类数量)'
            },
            {
                'name': 'covariance_type',
                'type': 'select',
                'default': 'full',
                'options': ['full', 'tied', 'diag', 'spherical'],
                'description': '协方差类型'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 100,
                'range': [50, 500],
                'description': '最大迭代次数'
            },
            {
                'name': 'n_init',
                'type': 'int',
                'default': 1,
                'range': [1, 20],
                'description': '初始化次数'
            },
            {
                'name': 'init_params',
                'type': 'select',
                'default': 'kmeans',
                'options': ['kmeans', 'random'],
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
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'GMMClustering':
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
        self.model = SKGaussianMixture(**validated_params)
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测属于各聚类的概率
        
        Args:
            X: 测试特征
            
        Returns:
            聚类概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.predict_proba(X)
