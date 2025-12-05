"""
主成分分析降维
Principal Component Analysis
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.decomposition import PCA as SKPCA
from ..base_algorithm import BaseAlgorithm


class PCAReduction(BaseAlgorithm):
    """主成分分析降维算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "pca"
        self.algorithm_type = "dimensionality_reduction"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 2,
                'range': [1, 100],
                'description': '降维后的维度数量'
            },
            {
                'name': 'whiten',
                'type': 'bool',
                'default': False,
                'description': '是否白化(使各成分方差为1)'
            },
            {
                'name': 'svd_solver',
                'type': 'select',
                'default': 'auto',
                'options': ['auto', 'full', 'arpack', 'randomized'],
                'description': 'SVD求解器'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray = None, **params) -> 'PCAReduction':
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
        self.model = SKPCA(**validated_params)
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
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """获取每个主成分的解释方差比例"""
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.explained_variance_ratio_
