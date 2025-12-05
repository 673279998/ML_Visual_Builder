"""
CatBoost分类算法
CatBoost Classifier
"""
import numpy as np
from typing import Dict, Any, List, Optional
from catboost import CatBoostClassifier as CatBoostClassifierModel
from ..base_algorithm import BaseAlgorithm
from backend.config import DATA_DIR


class CatBoostClassifier(BaseAlgorithm):
    """CatBoost分类算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "catboost_classifier"
        self.algorithm_type = "classification"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'iterations',
                'type': 'int',
                'default': 100,
                'range': [10, 1000],
                'description': '提升树的数量'
            },
            {
                'name': 'learning_rate',
                'type': 'float',
                'default': 0.1,
                'range': [0.001, 1.0],
                'description': '学习率'
            },
            {
                'name': 'depth',
                'type': 'int',
                'default': 6,
                'range': [1, 16],
                'description': '树的深度'
            },
            {
                'name': 'l2_leaf_reg',
                'type': 'float',
                'default': 3.0,
                'range': [1.0, 10.0],
                'description': 'L2正则化系数'
            },
            {
                'name': 'border_count',
                'type': 'int',
                'default': 254,
                'range': [1, 255],
                'description': '数值特征的分割点数量'
            },
            {
                'name': 'bagging_temperature',
                'type': 'float',
                'default': 1.0,
                'range': [0.0, 10.0],
                'description': 'Bayesian bootstrap温度'
            },
            {
                'name': 'random_strength',
                'type': 'float',
                'default': 1.0,
                'range': [0.0, 10.0],
                'description': '分裂时的随机强度'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'CatBoostClassifier':
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
        
        # 设置 train_dir 到 data 目录
        if 'train_dir' not in validated_params:
            validated_params['train_dir'] = str(DATA_DIR / 'catboost_info')
            
        # 创建并训练模型
        self.model = CatBoostClassifierModel(**validated_params, verbose=False)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return CatBoostClassifierModel()
    
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
