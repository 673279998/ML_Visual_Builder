"""
Hash编码器
Hash Encoding (Feature Hashing)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.feature_extraction import FeatureHasher
from .base_encoder import BaseEncoder


class HashEncoder(BaseEncoder):
    """Hash编码器 - 使用哈希函数编码"""
    
    def __init__(self, n_features: int = 8):
        """
        初始化
        
        Args:
            n_features: 哈希特征的数量
        """
        super().__init__()
        self.encoder_name = "hash"
        self.encoder_type = "categorical"
        self.n_features = n_features
        self.hasher = None
        self.feature_names = []
    
    def fit(self, data: pd.Series) -> 'HashEncoder':
        """拟合编码器"""
        # 创建FeatureHasher
        self.hasher = FeatureHasher(n_features=self.n_features, input_type='string')
        
        # 生成特征名称
        self.feature_names = [f"{data.name}_hash_{i}" for i in range(self.n_features)]
        
        # 建立编码映射（为测试兼容性）
        unique_categories = data.unique()
        for i, category in enumerate(unique_categories):
            self.encoding_map[str(category)] = i
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 转换为字符串列表
        data_str = [[str(x)] for x in data.values]
        
        # 哈希编码
        hashed = self.hasher.transform(data_str).toarray()
        
        # 转换为DataFrame
        result = pd.DataFrame(
            hashed,
            columns=self.feature_names,
            index=data.index
        )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """Hash编码不支持反向转换"""
        raise NotImplementedError("Hash编码不支持反向转换(哈希函数不可逆)")
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.feature_names
