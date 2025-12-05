"""
数据编码器基类
Base Encoder for data encoding
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class BaseEncoder(ABC):
    """编码器基类"""
    
    def __init__(self):
        self.encoder_name = ""
        self.encoder_type = ""
        self.fitted = False
        self.encoding_map = {}
    
    @abstractmethod
    def fit(self, data: pd.Series) -> 'BaseEncoder':
        """
        拟合编码器
        
        Args:
            data: 待编码的数据
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        转换数据
        
        Args:
            data: 待转换的数据
            
        Returns:
            转换后的数据
        """
        pass
    
    def fit_transform(self, data: pd.Series) -> pd.DataFrame:
        """
        拟合并转换数据
        
        Args:
            data: 待编码的数据
            
        Returns:
            转换后的数据
        """
        self.fit(data)
        return self.transform(data)
    
    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """
        反向转换
        
        Args:
            data: 编码后的数据
            
        Returns:
            原始数据
        """
        pass
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """获取编码信息"""
        return {
            'encoder_name': self.encoder_name,
            'encoder_type': self.encoder_type,
            'fitted': self.fitted,
            'encoding_map': self.encoding_map
        }
