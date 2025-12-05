"""
Target编码器
Target Encoding (Mean Encoding)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_encoder import BaseEncoder


class TargetEncoder(BaseEncoder):
    """Target编码器 - 使用目标变量的均值编码"""
    
    def __init__(self, smoothing: float = 0.0):
        """
        初始化
        
        Args:
            smoothing: 平滑参数,用于处理稀有类别(默认为0，即不使用平滑)
        """
        super().__init__()
        self.encoder_name = "target"
        self.encoder_type = "categorical"
        self.smoothing = smoothing
        self.global_mean = 0.0
        self.category_means = {}
        self.category_counts = {}
    
    def fit(self, data: pd.Series, target: pd.Series = None) -> 'TargetEncoder':
        """
        拟合编码器
        
        Args:
            data: 待编码的数据
            target: 目标变量(必需)
        """
        if target is None:
            raise ValueError("Target编码需要目标变量")
        
        # 计算全局均值
        self.global_mean = target.mean()
        
        # 计算每个类别的均值和数量
        for category in data.unique():
            mask = data == category
            category_target = target[mask]
            
            self.category_counts[str(category)] = len(category_target)
            self.category_means[str(category)] = category_target.mean()
            
            # 应用平滑
            count = self.category_counts[str(category)]
            cat_mean = self.category_means[str(category)]
            smoothed_mean = (count * cat_mean + self.smoothing * self.global_mean) / (count + self.smoothing)
            
            self.encoding_map[str(category)] = smoothed_mean
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 转换数据
        encoded = data.map(lambda x: self.encoding_map.get(str(x), self.global_mean))
        
        # 转换为DataFrame
        result = pd.DataFrame(
            {data.name: encoded},
            index=data.index
        )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """Target编码不支持反向转换"""
        raise NotImplementedError("Target编码不支持反向转换")
    
    def fit_transform(self, data: pd.Series, target: pd.Series = None) -> pd.DataFrame:
        """
        拟合并转换数据
        
        Args:
            data: 待编码的数据
            target: 目标变量(必需)
            
        Returns:
            转换后的数据
        """
        self.fit(data, target)
        return self.transform(data)
