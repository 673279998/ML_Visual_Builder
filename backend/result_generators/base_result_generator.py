"""
结果生成器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseResultGenerator(ABC):
    """结果生成器基类"""
    
    def __init__(self):
        self.result_type = ""
    
    @abstractmethod
    def generate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成性能指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model: 训练好的模型
            X_test: 测试特征数据
            
        Returns:
            包含性能指标的字典
        """
        pass
    
    @abstractmethod
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成可视化数据
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model: 训练好的模型
            X_test: 测试特征数据
            
        Returns:
            包含可视化数据的字典
        """
        pass
    
    def generate_algorithm_specific_info(self, model: Any, algorithm_name: str = None) -> Dict[str, Any]:
        """
        生成算法特定的额外信息
        
        Args:
            model: 训练好的模型
            algorithm_name: 算法名称
            
        Returns:
            包含算法特定信息的字典
        """
        # 默认返回空字典,子类可以重写
        return {}
    
    def generate_complete_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model: Any, X_test: np.ndarray = None,
                                 feature_names: List[str] = None,
                                 algorithm_name: str = None) -> Dict[str, Any]:
        """
        生成完整的结果(指标+可视化)
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model: 训练好的模型
            X_test: 测试特征数据
            feature_names: 特征名称列表
            
        Returns:
            包含完整结果的字典
        """
        results = {
            'metrics': self.generate_metrics(y_true, y_pred, model, X_test),
            'visualizations': self.generate_visualizations(y_true, y_pred, model, X_test),
            'result_type': self.result_type
        }
        
        # 添加特征名称(如果提供)
        if feature_names:
            results['feature_names'] = feature_names
        
        # 添加算法特定信息
        algorithm_info = self.generate_algorithm_specific_info(model, algorithm_name)
        if algorithm_info:
            results['algorithm_specific_info'] = algorithm_info
        
        return results
