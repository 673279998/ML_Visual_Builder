"""
梯度提升回归算法
Gradient Boosting Regressor
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import GradientBoostingRegressor as SKGradientBoostingRegressor
from ..base_algorithm import BaseAlgorithm


class GradientBoostingRegressor(BaseAlgorithm):
    """梯度提升回归算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "gradient_boosting_regressor"
        self.algorithm_type = "regression"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'n_estimators',
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
                'name': 'max_depth',
                'type': 'int',
                'default': 3,
                'range': [1, 20],
                'description': '每棵树的最大深度'
            },
            {
                'name': 'min_samples_split',
                'type': 'int',
                'default': 2,
                'range': [2, 100],
                'description': '分裂内部节点所需的最小样本数'
            },
            {
                'name': 'min_samples_leaf',
                'type': 'int',
                'default': 1,
                'range': [1, 50],
                'description': '叶子节点所需的最小样本数'
            },
            {
                'name': 'subsample',
                'type': 'float',
                'default': 1.0,
                'range': [0.1, 1.0],
                'description': '用于训练每棵树的样本比例'
            },
            {
                'name': 'max_features',
                'type': 'select',
                'default': 'None',
                'options': ['sqrt', 'log2', 'None'],
                'description': '寻找最佳分割时考虑的特征数量'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'GradientBoostingRegressor':
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
        
        # 处理max_features参数
        if validated_params.get('max_features') == 'None':
            validated_params['max_features'] = None
        
        # 创建并训练模型
        self.model = SKGradientBoostingRegressor(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKGradientBoostingRegressor()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 测试特征
            
        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用train()方法")
        
        return self.model.predict(X)
