"""
线性判别分析降维
Linear Discriminant Analysis
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLDA
from ..base_algorithm import BaseAlgorithm


class LDAReduction(BaseAlgorithm):
    """线性判别分析降维算法(需要标签)"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "lda"
        self.algorithm_type = "dimensionality_reduction"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 2,
                'range': [1, 100],
                'description': '降维后的维度数量(最多为类别数-1)'
            },
            {
                'name': 'solver',
                'type': 'select',
                'default': 'svd',
                'options': ['svd', 'lsqr', 'eigen'],
                'description': '求解器'
            },
            {
                'name': 'shrinkage',
                'type': 'select',
                'default': 'None',
                'options': ['None', 'auto'],
                'description': '收缩参数(仅lsqr和eigen支持)'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'LDAReduction':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 标签(LDA需要标签)
            **params: 超参数
            
        Returns:
            self
        """
        if y is None:
            raise ValueError("LDA降维需要标签,请提供y参数")
        
        # 验证超参数
        validated_params = self.validate_hyperparameters(params)
        
        # 处理shrinkage参数
        if validated_params.get('shrinkage') == 'None':
            validated_params['shrinkage'] = None
        
        # svd求解器不支持shrinkage
        if validated_params.get('solver') == 'svd' and validated_params.get('shrinkage') is not None:
            validated_params.pop('shrinkage', None)
        
        # 创建并训练模型
        self.model = SKLDA(**validated_params)
        self.model.fit(X, y)
        
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
