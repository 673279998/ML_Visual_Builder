"""
XGBoost回归算法
XGBoost Regressor
"""
import numpy as np
from typing import Dict, Any, List
from xgboost import XGBRegressor as XGBRegressorModel
from ..base_algorithm import BaseAlgorithm


class XGBoostRegressor(BaseAlgorithm):
    """XGBoost回归算法"""
    
    def __init__(self):
        super().__init__()
        self.algorithm_name = "xgboost_regressor"
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
                'default': 6,
                'range': [1, 20],
                'description': '树的最大深度'
            },
            {
                'name': 'min_child_weight',
                'type': 'int',
                'default': 1,
                'range': [1, 10],
                'description': '子节点最小权重和'
            },
            {
                'name': 'subsample',
                'type': 'float',
                'default': 1.0,
                'range': [0.1, 1.0],
                'description': '训练样本采样比例'
            },
            {
                'name': 'colsample_bytree',
                'type': 'float',
                'default': 1.0,
                'range': [0.1, 1.0],
                'description': '每棵树的特征采样比例'
            },
            {
                'name': 'gamma',
                'type': 'float',
                'default': 0.0,
                'range': [0.0, 10.0],
                'description': '节点分裂所需的最小损失减少'
            },
            {
                'name': 'reg_alpha',
                'type': 'float',
                'default': 0.0,
                'range': [0.0, 10.0],
                'description': 'L1正则化项'
            },
            {
                'name': 'reg_lambda',
                'type': 'float',
                'default': 1.0,
                'range': [0.0, 10.0],
                'description': 'L2正则化项'
            },
            {
                'name': 'random_state',
                'type': 'int',
                'default': 42,
                'range': [0, 10000],
                'description': '随机种子'
            }
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, **params) -> 'XGBoostRegressor':
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
        self.model = XGBRegressorModel(**validated_params)
        self.model.fit(X, y)
        
        return self
    
    def get_estimator(self) -> Any:
        """获取底层Scikit-learn估计器实例"""
        return XGBRegressorModel()
    
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
