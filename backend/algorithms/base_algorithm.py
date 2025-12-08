"""
算法基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


import inspect

class BaseAlgorithm(ABC):
    """算法基类,所有算法都继承此类"""
    
    def __init__(self):
        self.model = None
        self.algorithm_name = ""
        self.algorithm_type = ""  # classification, regression, clustering, reduction
        self.display_name = ""  # 中文显示名称
    
    def get_estimator(self) -> Any:
        """
        获取底层Scikit-learn估计器实例 (子类可覆盖)
        """
        return None

    @abstractmethod
    def get_hyperparameters(self) -> List[Dict[str, Any]]:
        """
        获取算法的超参数定义
        
        Returns:
            超参数定义列表,每个参数包含:
            - name: 参数名称
            - type: 参数类型(integer/float/categorical/boolean)
            - default: 默认值
            - range: 取值范围(对于数值型)
            - options: 可选值列表(对于分类型)
            - nullable: 是否可为空
            - description: 参数描述
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **params) -> 'BaseAlgorithm':
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标变量(对于监督学习)
            **params: 超参数
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        执行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        pass
    
    def validate_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证超参数有效性
        
        Args:
            params: 用户提供的超参数
            
        Returns:
            验证后的超参数
        """
        # print(f"DEBUG: validate_hyperparameters called with params: {params}")
        hyperparams_def = {hp['name']: hp for hp in self.get_hyperparameters()}
        validated = {}
        
        # 1. 处理所有用户提供的参数
        for name, value in params.items():
            if name in hyperparams_def:
                # 如果在定义中，进行类型和范围验证
                hp_def = hyperparams_def[name]
                hp_type = hp_def['type']
                
                # 类型转换
                try:
                    if hp_type == 'int':
                        value = int(value)
                    elif hp_type == 'float':
                        value = float(value)
                    elif hp_type == 'bool':
                        if isinstance(value, str):
                            value = value.lower() == 'true'
                        else:
                            value = bool(value)
                except (ValueError, TypeError):
                    # 类型转换失败，保留原值或使用默认值
                    print(f"WARNING: Parameter {name} type conversion failed, using original value: {value}")
                
                # 范围检查 (可选，如果转换成功)
                if 'range' in hp_def and isinstance(value, (int, float)):
                    min_val, max_val = hp_def['range']
                    if value < min_val:
                        value = min_val
                    elif value > max_val:
                        value = max_val
                
                validated[name] = value
            else:
                # 如果不在定义中，直接透传 (支持"所有参数")
                # print(f"DEBUG: Parameter {name} not in definition, passing through.")
                validated[name] = value
        
        # 2. 填充未提供的默认参数 (仅针对定义中存在的参数)
        for name, hp_def in hyperparams_def.items():
            if name not in validated:
                validated[name] = hp_def.get('default')
        
        # 3. 验证参数是否被底层模型接受 (如果可能)
        estimator = self.get_estimator()
        if estimator is not None:
            try:
                # 获取estimator.__init__的参数签名
                sig = inspect.signature(estimator.__init__)
                valid_params = set(sig.parameters.keys())
                
                # 检查所有validated中的参数是否在valid_params中
                # 注意: 需要排除 **kwargs 形式的参数接收者，如果有的话
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                
                if not has_kwargs:
                    invalid_params = []
                    for name in validated.keys():
                        if name not in valid_params:
                            invalid_params.append(name)
                    
                    if invalid_params:
                        error_msg = f"参数错误: 算法 {self.algorithm_name} 不支持以下参数: {', '.join(invalid_params)}。" \
                                    f"请检查拼写或查阅算法文档。"
                        # print(f"ERROR: {error_msg}")
                        raise ValueError(error_msg)
                        
            except ValueError as e:
                # 显式抛出的ValueError直接向上抛出
                raise e
            except Exception as e:
                # 如果验证过程中出错(例如无法获取签名)，则记录日志但不阻断，交由底层模型运行时抛出异常
                print(f"WARNING: Failed to validate parameters against estimator signature: {e}")

        return validated
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        获取算法的默认超参数
        
        Returns:
            默认超参数字典
        """
        hyperparams_def = self.get_hyperparameters()
        return {hp['name']: hp['default'] for hp in hyperparams_def if 'default' in hp}
    
    def get_estimator(self) -> Any:
        """
        获取底层Scikit-learn估计器实例(用于网格搜索等)
        
        Returns:
            Scikit-learn估计器实例或None
        """
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        返回模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'algorithm_name': self.algorithm_name,
            'algorithm_type': self.algorithm_type,
            'model_class': self.model.__class__.__name__ if self.model else None
        }
        
        # 尝试获取模型参数
        if self.model and hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性(如果模型支持)
        
        Returns:
            特征重要性字典或None
        """
        if self.model is None:
            return None
        
        # 检查模型是否有feature_importances_属性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        
        # 检查模型是否有coef_属性(如线性模型)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) == 1:
                return {f'feature_{i}': float(abs(c)) for i, c in enumerate(coef)}
            else:
                # 多类分类,取平均
                avg_coef = np.mean(np.abs(coef), axis=0)
                return {f'feature_{i}': float(c) for i, c in enumerate(avg_coef)}
        
        return None
