"""
Binary编码器
Binary Encoding
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .base_encoder import BaseEncoder


class BinaryEncoder(BaseEncoder):
    """Binary编码器 - 将整数标签转换为二进制表示"""
    
    def __init__(self):
        super().__init__()
        self.encoder_name = "binary"
        self.encoder_type = "categorical"
        self.categories = []
        self.n_bits = 0
        self.feature_names = []
    
    def fit(self, data: pd.Series) -> 'BinaryEncoder':
        """拟合编码器"""
        # 获取唯一类别
        self.categories = sorted(data.unique().tolist())
        
        # 计算需要的二进制位数
        n_categories = len(self.categories)
        self.n_bits = int(np.ceil(np.log2(n_categories))) if n_categories > 0 else 1
        
        # 保存编码映射
        for idx, cat in enumerate(self.categories):
            self.encoding_map[str(cat)] = idx
        
        # 生成特征名称
        self.feature_names = [f"{data.name}_bin_{i}" for i in range(self.n_bits)]
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """转换数据"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 映射到整数索引
        indices = data.map(lambda x: self.encoding_map.get(str(x), 0))
        
        # 转换为二进制
        binary_matrix = np.zeros((len(data), self.n_bits))
        for i, idx in enumerate(indices):
            binary_rep = format(int(idx), f'0{self.n_bits}b')
            binary_matrix[i] = [int(b) for b in binary_rep]
        
        # 转换为DataFrame
        result = pd.DataFrame(
            binary_matrix,
            columns=self.feature_names,
            index=data.index
        )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """反向转换"""
        if not self.fitted:
            raise ValueError("编码器尚未拟合,请先调用fit()方法")
        
        # 从二进制转换为整数索引
        indices = []
        for _, row in data.iterrows():
            binary_str = ''.join([str(int(b)) for b in row.values])
            idx = int(binary_str, 2)
            indices.append(idx)
        
        # 映射回原始类别
        decoded = [self.categories[idx] if idx < len(self.categories) else self.categories[0] 
                   for idx in indices]
        
        return pd.Series(decoded, index=data.index)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.feature_names
