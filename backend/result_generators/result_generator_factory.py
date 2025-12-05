"""
结果生成器工厂类
"""
from typing import Dict, Any
from .base_result_generator import BaseResultGenerator
from .classification_result_generator import ClassificationResultGenerator
from .regression_result_generator import RegressionResultGenerator
from .clustering_result_generator import ClusteringResultGenerator
from .dimensionality_reduction_result_generator import DimensionalityReductionResultGenerator


class ResultGeneratorFactory:
    """结果生成器工厂类"""
    
    _generators = {
        'classification': ClassificationResultGenerator,
        'regression': RegressionResultGenerator,
        'clustering': ClusteringResultGenerator,
        'dimensionality_reduction': DimensionalityReductionResultGenerator
    }
    
    @classmethod
    def create_generator(cls, algorithm_type: str) -> BaseResultGenerator:
        """
        创建结果生成器实例
        
        Args:
            algorithm_type: 算法类型
            
        Returns:
            结果生成器实例
            
        Raises:
            ValueError: 不支持的算法类型
        """
        if algorithm_type not in cls._generators:
            raise ValueError(f"不支持的算法类型: {algorithm_type}")
        
        return cls._generators[algorithm_type]()
    
    @classmethod
    def get_supported_types(cls) -> list:
        """获取支持的算法类型列表"""
        return list(cls._generators.keys())
    
    @classmethod
    def register_generator(cls, algorithm_type: str, generator_class: type):
        """
        注册新的结果生成器
        
        Args:
            algorithm_type: 算法类型
            generator_class: 生成器类
        """
        cls._generators[algorithm_type] = generator_class
