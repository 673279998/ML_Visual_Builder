"""
决策树回归算法
Decision Tree Regressor
"""
import numpy as np
from typing import Dict, Any, List
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from ..base_algorithm import BaseAlgorithm


class DecisionTreeRegressor(BaseAlgorithm):
    """决策树回归算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "decision_tree_regressor"
        self.algorithm_type = "regression"
    
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """获取超参数定义"""
        return [
            {
                'name': 'criterion',
                'type': 'select',
                'default': 'squared_error',
                'options': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'description': '节点分裂准则'
            },
            {
                'name': 'max_depth',
                'type': 'int',
                'default': None,
                'range': [1, 50],
                'description': '树的最大深度,None表示无限制'
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
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'DecisionTreeRegressor':
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
        
        # 处理None值参数
        if validated_params.get('max_features') == 'None':
            validated_params['max_features'] = None
        if validated_params.get('max_depth') is None:
            validated_params.pop('max_depth', None)
        
        # 创建并训练模型
        self.model = SKDecisionTreeRegressor(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return SKDecisionTreeRegressor()

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
