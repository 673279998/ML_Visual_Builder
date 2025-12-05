"""
支持向量机分类算法
SVM Classifier
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.svm import SVC
from ..base_algorithm import BaseAlgorithm


class SVMClassifier(BaseAlgorithm):
    """支持向量机分类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "svm_classifier"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'C',
                'type': 'float',
                'default': 1.0,
                'range': [0.001, 100.0],
                'description': '正则化参数,值越小正则化越强'
            },
            {
                'name': 'kernel',
                'type': 'select',
                'default': 'rbf',
                'options': ['linear', 'poly', 'rbf', 'sigmoid'],
                'description': '核函数类型'
            },
            {
                'name': 'degree',
                'type': 'int',
                'default': 3,
                'range': [2, 10],
                'description': '多项式核函数的度数(仅对poly核有效)'
            },
            {
                'name': 'gamma',
                'type': 'select',
                'default': 'scale',
                'options': ['scale', 'auto'],
                'description': '核函数系数,scale=1/(n_features*X.var()),auto=1/n_features'
            },
            {
                'name': 'probability',
                'type': 'bool',
                'default': True,
                'description': '是否启用概率估计'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'SVMClassifier':
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
        self.model = SVC(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SVC()
    
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
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        预测概率
        
        Args:
            X: 测试特征
            
        Returns:
            预测概率(如果启用了probability参数)
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        if hasattr(self.model, 'probability') and self.model.probability:
            return self.model.predict_proba(X)
        else:
            return None
