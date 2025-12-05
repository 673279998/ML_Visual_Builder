"""
算法基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class BaseAlgorithm(ABC):
    """算法基类,所有算法都继承此类"""
    
    def __init__(self):
        self.model = None
        self.algorithm_name = ""
        self.algorithm_type = ""  # classification, regression, clustering, reduction
    
    @abstractmethod
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """
        获取算法的超参数定义
        
        Returns:
            超参数定义列表,每个参数包含:
            - name: 参数名称
            - type: 参数类型(integer/float/categorical/boolean)
            - default: 默认值
            - range: 取值范围(对于数值型)
            - options: 可选值列表(对于分类型)
            - nullable: 是否可为空
            - description: 参数描述
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **params) -> 'BaseAlgorithm':
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标变量(对于监督学习)
            **params: 超参数
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        执行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        pass
    
    def validate_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证超参数有效性
        
        Args:
            params: 用户提供的超参数
            
        Returns:
            验证后的超参数
        """
        hyperparams_def = {hp['name']: hp for hp in self.get_hyperparameters()}
        validated = {}
        
        for name, value in params.items():
            if name not in hyperparams_def:
                continue  # 忽略未定义的参数
            
            hp_def = hyperparams_def[name]
            hp_type = hp_def['type']
            
            # 类型验证
            if hp_type == 'integer':
                value = int(value) if value is not None else hp_def.get('default')
                if value is not None and 'range' in hp_def:
                    min_val, max_val = hp_def['range']
                    value = max(min_val, min(max_val, value))
            elif hp_type == 'float':
                value = float(value) if value is not None else hp_def.get('default')
                if value is not None and 'range' in hp_def:
                    min_val, max_val = hp_def['range']
                    value = max(min_val, min(max_val, value))
            elif hp_type == 'categorical':
                if value not in hp_def.get('options', []):
                    value = hp_def.get('default')
            elif hp_type == 'boolean':
                value = bool(value) if value is not None else hp_def.get('default')
            
            validated[name] = value
        
        # 添加未提供的参数的默认值
        for name, hp_def in hyperparams_def.items():
            if name not in validated:
                validated[name] = hp_def.get('default')
        
        return validated
    
    def get_estimator(self) -> Any:
        """
        获取底层Scikit-learn估计器实例(用于网格搜索等)
        
        Returns:
            Scikit-learn估计器实例或None
        """
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        返回模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'algorithm_name': self.algorithm_name,
            'algorithm_type': self.algorithm_type,
            'model_class': self.model.__class__.__name__ if self.model else None
        }
        
        # 尝试获取模型参数
        if self.model and hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性(如果模型支持)
        
        Returns:
            特征重要性字典或None
        """
        if self.model is None:
            return None
        
        # 检查模型是否有feature_importances_属性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        
        # 检查模型是否有coef_属性(如线性模型)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) == 1:
                return {f'feature_{i}': float(abs(c)) for i, c in enumerate(coef)}
            else:
                # 多类分类,取平均
                avg_coef = np.mean(np.abs(coef), axis=0)
                return {f'feature_{i}': float(c) for i, c in enumerate(avg_coef)}
        
        return None
