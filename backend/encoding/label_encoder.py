"""
Label编码器
Label Encoding
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
from .base_encoder import BaseEncoder


class LabelEncoder(BaseEncoder):
    """Label编码器 - 将类别映射为整数"""
    
    def __init__(self):
        super().__init__()
        self.encoder_name = "label"
        self.encoder_type = "categorical"
        self.encoder = None
        self.classes = []
    
    def fit(self, data: pd.Series) -> 'LabelEncoder':
        """拟合编码器"""
        # 创建sklearn的LabelEncoder
        self.encoder = SKLabelEncoder()
        
        # 拟合数据
        self.encoder.fit(data)
        
        # 保存类别信息
        self.classes = self.encoder.classes_.tolist()
        
        # 保存编码映射
        for idx, cls in enumerate(self.classes):
            self.encoding_map[str(cls)] = idx
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 转换数据
        encoded = self.encoder.transform(data)
        
        # 转换为DataFrame
        result = pd.DataFrame(
            {data.name: encoded},
            index=data.index
        )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """反向转换"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 反向转换
        decoded = self.encoder.inverse_transform(data.values.flatten().astype(int))
        
        return pd.Series(decoded, index=data.index)
