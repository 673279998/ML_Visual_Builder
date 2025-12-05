"""
Frequency编码器
Frequency Encoding
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_encoder import BaseEncoder


class FrequencyEncoder(BaseEncoder):
    """Frequency编码器 - 使用类别出现频率编码"""
    
    def __init__(self, normalize: bool = False):
        """
        初始化
        
        Args:
            normalize: 是否归一化频率(True为比例,False为计数)
        """
        super().__init__()
        self.encoder_name = "frequency"
        self.encoder_type = "categorical"
        self.normalize = normalize
        self.frequency_map = {}
        self.total_count = 0
    
    def fit(self, data: pd.Series) -> 'FrequencyEncoder':
        """拟合编码器"""
        # 计算频率
        value_counts = data.value_counts()
        self.total_count = len(data)
        
        # 保存频率映射
        for category, count in value_counts.items():
            if self.normalize:
                self.frequency_map[str(category)] = count / self.total_count
            else:
                self.frequency_map[str(category)] = count
            
            self.encoding_map[str(category)] = self.frequency_map[str(category)]
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 转换数据,未知类别使用0
        encoded = data.map(lambda x: self.frequency_map.get(str(x), 0))
        
        # 转换为DataFrame
        result = pd.DataFrame(
            {data.name: encoded},
            index=data.index
        )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """Frequency编码不支持反向转换"""
        raise NotImplementedError("Frequency编码不支持反向转换")
