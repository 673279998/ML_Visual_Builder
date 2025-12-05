"""
Ordinal编码器
Ordinal Encoding
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import OrdinalEncoder as SKOrdinalEncoder
from .base_encoder import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    """Ordinal编码器 - 保持类别顺序的编码"""
    
    def __init__(self, categories: Optional[List] = None):
        """
        初始化
        
        Args:
            categories: 指定的类别顺序,如果为None则自动推断
        """
        super().__init__()
        self.encoder_name = "ordinal"
        self.encoder_type = "categorical"
        self.encoder = None
        self.categories = categories
        self.fitted_categories = []
    
    def fit(self, data: pd.Series) -> 'OrdinalEncoder':
        """拟合编码器"""
        # 创建sklearn的OrdinalEncoder
        if self.categories is not None:
            self.encoder = SKOrdinalEncoder(categories=[self.categories])
        else:
            self.encoder = SKOrdinalEncoder()
        
        # 拟合数据
        data_reshaped = data.values.reshape(-1, 1)
        self.encoder.fit(data_reshaped)
        
        # 保存类别信息
        self.fitted_categories = self.encoder.categories_[0].tolist()
        
        # 保存编码映射
        for idx, cat in enumerate(self.fitted_categories):
            self.encoding_map[str(cat)] = idx
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 转换数据
        data_reshaped = data.values.reshape(-1, 1)
        encoded = self.encoder.transform(data_reshaped)
        
        # 转换为DataFrame
        result = pd.DataFrame(
            {data.name: encoded.flatten()},
            index=data.index
        )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """反向转换"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 反向转换
        decoded = self.encoder.inverse_transform(data.values)
        
        return pd.Series(decoded.flatten(), index=data.index)
