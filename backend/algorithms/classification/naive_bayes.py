"""
朴素贝叶斯分类算法
Naive Bayes Classifier
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.naive_bayes import GaussianNB
from ..base_algorithm import BaseAlgorithm


class NaiveBayesClassifier(BaseAlgorithm):
    """朴素贝叶斯分类算法(高斯)"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "naive_bayes"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'var_smoothing',
                'type': 'float',
                'default': 1e-9,
                'range': [1e-12, 1e-5],
                'description': '为稳定性而添加到方差的部分'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'NaiveBayesClassifier':
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
        self.model = GaussianNB(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return GaussianNB()
    
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
