"""
One-Hot编码器
One-Hot Encoding
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder
from .base_encoder import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """One-Hot编码器"""
    
    def __init__(self, handle_unknown: str = 'ignore', sparse: bool = False):
        """
        初始化
        
        Args:
            handle_unknown: 未知类别处理方式('error'或'ignore')
            sparse: 是否返回稀疏矩阵
        """
        super().__init__()
        self.encoder_name = "one_hot"
        self.encoder_type = "categorical"
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.encoder = None
        self.categories = []
        self.feature_names = []
    
    def fit(self, data: pd.Series) -> 'OneHotEncoder':
        """拟合编码器"""
        # 创建sklearn的OneHotEncoder
        self.encoder = SKOneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse
        )
        
        # 拟合数据
        data_reshaped = data.values.reshape(-1, 1)
        self.encoder.fit(data_reshaped)
        
        # 保存类别信息
        self.categories = self.encoder.categories_[0].tolist()
        self.feature_names = [f"{data.name}_{cat}" for cat in self.categories]
        
        # 保存编码映射
        for idx, cat in enumerate(self.categories):
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
        if self.sparse:
            encoded = encoded.toarray()
        
        result = pd.DataFrame(
            encoded,
            columns=self.feature_names,
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
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.feature_names
