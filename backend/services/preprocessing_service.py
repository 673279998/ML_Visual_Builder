"""
预处理服务
Preprocessing Service
"""
import numpy as np
import pandas as pd
import joblib
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from backend.encoding.label_encoder import LabelEncoder
from backend.encoding.one_hot_encoder import OneHotEncoder
from backend.encoding.ordinal_encoder import OrdinalEncoder
from backend.encoding.target_encoder import TargetEncoder
from backend.encoding.frequency_encoder import FrequencyEncoder
from backend.encoding.binary_encoder import BinaryEncoder
from backend.encoding.hash_encoder import HashEncoder
from backend.config import PREPROCESSOR_DIR, ENCODER_DIR
import logging

logger = logging.getLogger(__name__)


class PreprocessingService:
    """数据预处理服务"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.preprocessing_history = []
        
    def _save_object(self, obj: Any, category: str, dataset_id: Optional[int], name_suffix: str, workflow_id: Optional[str] = None) -> Optional[str]:
        """
        保存预处理对象到磁盘
        
        Args:
            obj: 要保存的对象
            category: 类别 ('imputer', 'scaler', 'encoder')
            dataset_id: 数据集ID
            name_suffix: 文件名后缀 (如列名或方法名)
            workflow_id: 工作流ID (可选，用于隔离不同工作流的组件)
            
        Returns:
            保存的文件路径
        """
        try:
            # 确定基础保存目录
            if category == 'encoder':
                base_dir = ENCODER_DIR
            else:
                base_dir = PREPROCESSOR_DIR
                
            # 路径策略调整：
            # 1. 如果有 workflow_id，优先使用 workflow_id 作为主目录，实现"一次任务一个文件夹"
            # 2. 如果没有 workflow_id，回退到使用 dataset_id (保持兼容性)
            
            if workflow_id:
                # 结构: data/encoders/{workflow_id}/{filename}
                # 结构: data/preprocessors/{workflow_id}/{filename}
                save_dir = base_dir / str(workflow_id)
            elif dataset_id is not None:
                # 结构: data/encoders/{dataset_id}/{filename}
                save_dir = base_dir / str(dataset_id)
            else:
                # 既无workflow_id也无dataset_id，保存到common目录
                save_dir = base_dir / 'common'
                
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 生成文件名: {category}_{suffix}_{timestamp}.pkl
            timestamp = int(time.time())
            filename = f"{category}_{name_suffix}_{timestamp}.pkl"
            filepath = os.path.join(save_dir, filename)
            
            # 保存
            joblib.dump(obj, filepath)
            logger.info(f"已保存 {category} 对象到: {filepath}")
            
            # 返回相对路径，便于数据库存储和跨平台使用
            # 相对于项目根目录的路径
            try:
                from backend.config import BASE_DIR
                rel_path = os.path.relpath(filepath, BASE_DIR)
                # 转换为正斜杠，确保跨平台兼容性
                rel_path = rel_path.replace('\\', '/')
                return rel_path
            except:
                # 如果计算相对路径失败，返回绝对路径
                return str(filepath)
            
        except Exception as e:
            logger.error(f"保存 {category} 对象失败: {str(e)}")
            return None

    def _handle_datetime_columns(self, df: pd.DataFrame, info: Dict[str, Any] = None, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        自动检测并转换日期列为时间戳(float)
        """
        for col in df.columns:
            # 如果明确指定为分类变量，则跳过日期检测
            if categorical_columns and col in categorical_columns:
                continue
                
            if df[col].dtype == 'object':
                try:
                    # 检查非空样本
                    valid_sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if valid_sample is None:
                        continue
                        
                    # 检查样本长度，如果过长（例如超过100字符），可能是垃圾数据或长文本，跳过日期检测
                    if isinstance(valid_sample, str) and len(valid_sample) > 100:
                        continue

                    # 简单的格式检查，避免将纯数字或仅由数字和点组成的字符串误判为日期
                    s = str(valid_sample)
                    if s.isdigit() or all(ch.isdigit() or ch == '.' for ch in s):
                        continue
                         
                    # 尝试转换
                    pd.to_datetime(valid_sample)
                    
                    # 批量转换
                    temp_series = pd.to_datetime(df[col], errors='coerce')
                    
                    # 如果大部分转换成功（例如 > 80% 非空值）
                    non_null_count = df[col].count()
                    if non_null_count > 0:
                        converted_count = temp_series.count()
                        if converted_count / non_null_count > 0.8:
                            # 转换为时间戳 (float)
                            # NaT -> NaN
                            df[col] = temp_series.map(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
                            logger.info(f"列 {col} 已转换为时间戳")
                            if info is not None:
                                if 'date_columns_converted' not in info:
                                    info['date_columns_converted'] = []
                                info['date_columns_converted'].append(col)
                            
                except Exception:
                    continue
        return df

    def _get_valid_numeric_columns(self, df: pd.DataFrame, columns: Optional[List[str]] = None, categorical_columns: Optional[List[str]] = None) -> List[str]:
        """
        获取有效的数值列，严格排除对象类型和无法转换为数值的列
        """
        # 1. 初步筛选数值类型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 2. 如果指定了列范围，取交集
        if columns is not None:
            numeric_cols = [c for c in numeric_cols if c in columns]
            
        # 3. 二次校验：确保排除 object 类型和明确指定的分类变量
        valid_cols = []
        # 规范化分类列名（去除空格）
        norm_categorical_cols = set()
        if categorical_columns:
            norm_categorical_cols = {c.strip() for c in categorical_columns}

        for col in numeric_cols:
            if df[col].dtype == 'object':
                continue
            
            # 检查列名是否在分类列表中 (忽略空格)
            if norm_categorical_cols and col.strip() in norm_categorical_cols:
                continue
                
            valid_cols.append(col)
            
        return valid_cols
    
    # ==================== 缺失值处理 ====================
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None,
        dataset_id: Optional[int] = None,
        workflow_id: Optional[str] = None,
        target_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理缺失值
        
        Args:
            data: 数据集
            strategy: 填充策略 ('mean', 'median', 'most_frequent', 'constant', 'drop')
            columns: 要处理的列,None表示所有列
            fill_value: strategy为'constant'时的填充值
            dataset_id: 数据集ID (用于保存填充器)
            target_columns: 目标变量列名列表，用于校验和排除
            categorical_columns: 分类变量列名列表，用于强制作为非数值处理
            
        Returns:
            处理后的数据和处理信息
        """
        df = data.copy()
        info = {
            'strategy': strategy,
            'columns_processed': [],
            'missing_counts_before': {},
            'missing_counts_after': {},
            'saved_imputers': []
        }
        
        if columns is None:
            columns = df.columns.tolist()

        # 规范化列名集合
        norm_target_cols = {c.strip() for c in target_columns} if target_columns else set()
        norm_categorical_cols = {c.strip() for c in categorical_columns} if categorical_columns else set()

        # 校验目标变量并从处理列表中排除
        if target_columns:
            for target_col in target_columns:
                if target_col in df.columns:
                    # 检查是否有缺失值
                    if df[target_col].isna().sum() > 0:
                        raise ValueError(f"目标变量 '{target_col}' 存在缺失值，请先手动处理目标变量的缺失值")
                    
        # 过滤columns：仅排除目标列 (分类列保留，但在处理时强制使用非数值策略)
        final_columns = []
        for col in columns:
            col_clean = col.strip()
            if col_clean in norm_target_cols:
                continue
            final_columns.append(col)
        columns = final_columns

        # 自动处理日期列
        try:
            df = self._handle_datetime_columns(df, info, categorical_columns)
        except Exception as e:
            logger.error(f"自动处理日期列失败: {str(e)}")

        # 记录处理前的缺失值数量（仅记录参与处理的列）
        for col in columns:
            if col in df.columns:  # 日期转换可能会改变列（虽然这里没有改变列名，但最好检查一下）
                info['missing_counts_before'][col] = int(df[col].isna().sum())
        
        if strategy == 'drop':
            # 删除包含缺失值的行
            df = df.dropna(subset=columns)
        else:
            # 使用imputer填充
            for col in columns:
                try:
                    if col not in df.columns:
                        continue
                    if df[col].isna().sum() == 0:
                        continue
                    if strategy == 'constant':
                        df[col].fillna(fill_value, inplace=True)
                        continue
                        
                    current_strategy = strategy
                    
                    # 检查是否为明确的分类列
                    col_clean = col.strip()
                    is_categorical = col_clean in norm_categorical_cols
                    
                    # 只有非分类列且类型为数值时，才视为数值列
                    is_numeric = (not is_categorical) and (df[col].dtype in ['int64', 'float64'])
                    
                    if is_numeric:
                        if strategy in ['knn', 'tree']:
                            numeric_cols = [c for c in columns if c in df.columns and df[c].dtype in ['int64', 'float64'] and df[c].isna().sum() > 0]
                            if categorical_columns:
                                numeric_cols = [c for c in numeric_cols if c not in categorical_columns]
                            if col not in numeric_cols:
                                continue
                            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                            if strategy == 'knn':
                                imputer = KNNImputer(n_neighbors=5)
                            else:
                                imputer = IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1), max_iter=10, random_state=0)
                            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                            self.imputers[f"{strategy}_numeric"] = imputer
                            filepath = self._save_object(imputer, 'imputer', dataset_id, f"{strategy}_numeric")
                            if filepath:
                                info['saved_imputers'].append({'path': filepath, 'columns': numeric_cols, 'strategy': strategy})
                            info['columns_processed'].extend(numeric_cols)
                            continue
                        elif strategy == 'mode':
                            current_strategy = 'most_frequent'
                        if current_strategy in ['mean', 'median', 'most_frequent']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            imputer = SimpleImputer(strategy=current_strategy)
                            df[col] = imputer.fit_transform(df[[col]]).ravel()
                            self.imputers[col] = imputer
                            filepath = self._save_object(imputer, 'imputer', dataset_id, f"{col}_{current_strategy}", workflow_id)
                            if filepath:
                                info['saved_imputers'].append({'path': filepath, 'columns': [col], 'strategy': current_strategy})
                            info['columns_processed'].append(col)
                    else:
                        if strategy in ['knn', 'tree'] or current_strategy == 'mode':
                            current_strategy = 'most_frequent'
                        if current_strategy in ['mean', 'median']:
                            current_strategy = 'most_frequent'
                        if current_strategy == 'most_frequent':
                            original_dtype = df[col].dtype
                            df[col] = df[col].astype(str)
                            df[col] = df[col].replace('nan', np.nan)
                            imputer = SimpleImputer(strategy='most_frequent')
                            filled_col = imputer.fit_transform(df[[col]]).ravel()
                            df[col] = filled_col
                            if original_dtype != 'object':
                                try:
                                    df[col] = df[col].astype(original_dtype)
                                except Exception:
                                    pass
                            self.imputers[col] = imputer
                            filepath = self._save_object(imputer, 'imputer', dataset_id, f"{col}_{current_strategy}", workflow_id)
                            if filepath:
                                info['saved_imputers'].append({'path': filepath, 'columns': [col], 'strategy': current_strategy})
                            info['columns_processed'].append(col)
                except Exception as e:
                    logger.error(f"处理列 {col} 缺失值失败: {str(e)}")
        
        # 记录处理后的缺失值数量
        for col in columns:
            if col in df.columns:
                info['missing_counts_after'][col] = int(df[col].isna().sum())
        
        self.preprocessing_history.append({
            'operation': 'handle_missing_values',
            'info': info
        })
        
        return df, info
    
    # ==================== 数据缩放 ====================
    
    def scale_features(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        columns: Optional[List[str]] = None,
        dataset_id: Optional[int] = None,
        workflow_id: Optional[str] = None,
        suffix: str = None,
        target_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        数据缩放/标准化
        
        Args:
            data: 数据集
            method: 缩放方法 ('standard', 'minmax', 'robust')
            columns: 要缩放的列,None表示所有数值列
            dataset_id: 数据集ID (用于保存缩放器)
            target_columns: 目标变量列名列表，需单独缩放
            categorical_columns: 分类变量列名列表，需排除
            
        Returns:
            缩放后的数据和缩放信息
        """
        df = data.copy()
        info = {
            'method': method,
            'columns_scaled': [],
            'scaler_params': {},
            'saved_scaler': None,
            'saved_scalers': []  # 支持多个scaler
        }
        
        # 自动处理日期列
        df = self._handle_datetime_columns(df, info, categorical_columns)
        
        # 1. 确定所有需要缩放的列
        if columns is None:
            columns = self._get_valid_numeric_columns(df, categorical_columns=categorical_columns)
        else:
            # 过滤掉非数值列
            valid_columns = self._get_valid_numeric_columns(df, columns, categorical_columns=categorical_columns)
            if len(valid_columns) < len(columns):
                invalid_cols = set(columns) - set(valid_columns)
                logger.warning(f"忽略非数值列的缩放请求: {invalid_cols}")
            columns = valid_columns

        # 2. 分离特征列和目标列
        feature_cols = []
        target_cols = []
        
        norm_target_cols = {c.strip() for c in target_columns} if target_columns else set()

        if target_columns:
            for col in columns:
                # 如果是目标列
                if col.strip() in norm_target_cols:
                    # 只有数值型的目标列才参与缩放
                    if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'object':
                        target_cols.append(col)
                else:
                    feature_cols.append(col)
        else:
            feature_cols = columns

        # 3. 辅助函数：执行缩放并保存
        def _apply_scaling(cols_to_scale, scaler_suffix, scaler_type_name='features'):
            if not cols_to_scale:
                return

            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"未知的缩放方法: {method}")

            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            
            # 保存scaler
            scaler_key = f"{method}_{scaler_suffix}"
            self.scalers[scaler_key] = scaler
            
            # 持久化
            save_suffix = f"{method}_{scaler_suffix}"
            if suffix:
                save_suffix = f"{save_suffix}_{suffix}"
                
            filepath = self._save_object(scaler, 'scaler', dataset_id, save_suffix, workflow_id)
            
            scaler_info = {
                'path': filepath,
                'columns': cols_to_scale,
                'method': method,
                'type': scaler_type_name 
            }
            
            if filepath:
                # 兼容旧格式（主要用于特征scaler）
                if scaler_type_name == 'features':
                    info['saved_scaler'] = scaler_info
                
                info['saved_scalers'].append(scaler_info)

            # 记录参数信息 (仅记录特征的，避免混淆，或者分别记录)
            if scaler_type_name == 'features':
                if method == 'standard':
                    info['scaler_params'] = {
                        'mean': scaler.mean_.tolist(),
                        'std': scaler.scale_.tolist()
                    }
                elif method == 'minmax':
                    info['scaler_params'] = {
                        'min': scaler.min_.tolist(),
                        'scale': scaler.scale_.tolist()
                    }

        # 4. 分别缩放特征和目标
        if feature_cols:
            _apply_scaling(feature_cols, 'features', 'features')
            info['columns_scaled'].extend(feature_cols)
            
        if target_cols:
            # 再次检查目标列是否为数值型，防止分类型目标被缩放
            valid_target_cols = []
            for col in target_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'object':
                     valid_target_cols.append(col)
                else:
                    logger.info(f"跳过目标列 {col} 的缩放，因为它不是数值型")
            
            if valid_target_cols:
                _apply_scaling(valid_target_cols, 'target', 'target')
                info['columns_scaled'].extend(valid_target_cols)
        
        self.preprocessing_history.append({
            'operation': 'scale_features',
            'info': info
        })
        
        return df, info

    # ==================== 自动编码 ====================

    def auto_encode_features(
        self,
        data: pd.DataFrame,
        column_configs: List[Dict[str, Any]],
        dataset_id: Optional[int] = None,
        workflow_id: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        自动编码特征
        
        Args:
            data: 数据集
            column_configs: 列配置列表
            dataset_id: 数据集ID (用于保存编码器)
            target_column: 目标列名 (用于Target Encoding)
            
        Returns:
            处理后的数据和处理信息
        """
        df = data.copy()
        info = {
            'operation': 'auto_encode',
            'columns_encoded': [],
            'encoders': {},
            'errors': [],
            'saved_encoders': []
        }
        
        # 自动处理日期列
        categorical_columns = [c['column_name'] for c in column_configs if c.get('data_type') == 'categorical']
        
        # 注意：_handle_datetime_columns 可能会修改 info (虽然目前没有)，
        # 但主要风险在于它可能引入一些非标准类型的统计信息
        df = self._handle_datetime_columns(df, info, categorical_columns=categorical_columns)
        
        encoder_map = {
            'label': LabelEncoder,
            'onehot': OneHotEncoder,
            'ordinal': OrdinalEncoder,
            'target': TargetEncoder,
            'frequency': FrequencyEncoder,
            'binary': BinaryEncoder,
            'hash': HashEncoder
        }
        
        # 寻找目标列 (用于Target Encoding)
        target_series = None
        
        # 优先使用传入的target_column
        if target_column and target_column in df.columns:
            target_series = df[target_column]
        
        # 如果未传入或未找到，尝试从column_configs中查找is_target标记
        if target_series is None:
            for col_config in column_configs:
                if col_config.get('is_target'):
                    found_target = col_config['column_name']
                    if found_target in df.columns:
                        target_series = df[found_target]
                        # 如果没有传入target_column，记录找到的目标列
                        if not target_column:
                            target_column = found_target
                    break
        
        # 构建配置字典，方便查找 (支持列名去除空格)
        config_map = {c['column_name'].strip(): c for c in column_configs}
        
        # 遍历所有列，决定编码方式
        # 1. 使用配置中的方式
        # 2. 如果未配置且为对象类型，默认使用onehot
        
        # 获取所有列名，包括配置中可能没有的
        all_columns = df.columns.tolist()
        
        for col_name in all_columns:
            # 获取该列的配置 (使用去除空格后的列名匹配)
            col_clean = col_name.strip()
            col_config = config_map.get(col_clean, {})
            
            encoding_method = col_config.get('encoding_method')
            is_excluded = col_config.get('is_excluded', False)
            
            # 如果是目标列，通常不做特征编码（除非是特定任务，但这里假设特征编码是针对特征的）
            # 不过有些场景下可能需要对目标列编码（如分类任务的LabelEncoder），这里暂不自动处理目标列
            if col_name == target_column:
                # 如果显式配置了编码方法，则继续；否则跳过
                if not encoding_method:
                    continue
            
            # 自动检测：如果是对象类型且未指定编码方法且未被排除
            if not encoding_method and not is_excluded:
                if df[col_name].dtype == 'object' or df[col_name].dtype.name == 'category':
                    logger.info(f"列 {col_name} 为对象类型且未配置编码，自动应用 One-Hot 编码")
                    encoding_method = 'onehot'
            
            # 跳过排除的列、无编码方法的列
            if is_excluded or not encoding_method or encoding_method == 'none':
                continue
                
            if encoding_method in encoder_map:
                try:
                    encoder_class = encoder_map[encoding_method]
                    encoder = encoder_class()
                    
                    # Target Encoding特殊处理
                    if encoding_method == 'target':
                        if target_series is None:
                            logger.warning(f"列 {col_name} 配置为Target编码，但未找到目标列，跳过编码")
                            info['errors'].append(f"列 {col_name}: 缺少目标列，无法进行Target编码")
                            continue
                        transformed_df = encoder.fit_transform(df[col_name], target_series)
                    else:
                        transformed_df = encoder.fit_transform(df[col_name])
                    
                    if encoding_method == 'onehot':
                        # OneHot会产生多列，删除原列，合并新列
                        df = df.drop(columns=[col_name])
                        # 确保索引对齐
                        transformed_df.index = df.index
                        df = pd.concat([df, transformed_df], axis=1)
                        new_cols = transformed_df.columns.tolist()
                    else:
                        # 其他编码替换原列
                        df[col_name] = transformed_df.iloc[:, 0]
                        new_cols = [col_name]
                        
                    info['columns_encoded'].append({
                        'column': col_name,
                        'method': encoding_method,
                        'new_columns': new_cols
                    })
                    
                    # 保存encoder
                    self.encoders[col_name] = encoder
                    filepath = self._save_object(encoder, 'encoder', dataset_id, f"{col_name}_{encoding_method}", workflow_id)
                    if filepath:
                        info['saved_encoders'].append({
                            'path': filepath,
                            'column': col_name,
                            'method': encoding_method
                        })
                    
                except Exception as e:
                    error_msg = f"列 {col_name} 使用 {encoding_method} 编码失败: {str(e)}"
                    logger.error(error_msg)
                    info['errors'].append(error_msg)
            else:
                logger.warning(f"未知的编码方法: {encoding_method} (列: {col_name})")
            
        self.preprocessing_history.append({
            'operation': 'auto_encode',
            'info': info
        })
        
        return df, info
    
    # ==================== 异常值处理 ====================
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'iqr',
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        检测异常值
        
        Args:
            data: 数据集
            method: 检测方法 ('iqr', 'zscore', 'isolation_forest')
            columns: 要检测的列,None表示所有数值列
            threshold: 阈值 (IQR方法为倍数,Z-score方法为标准差倍数)
            categorical_columns: 分类变量列名列表
            
        Returns:
            (原始数据, 异常值信息)
        """
        from sklearn.ensemble import IsolationForest

        outliers_info = {
            'method': method,
            'threshold': threshold,
            'outliers_by_column': {}
        }
        
        # 自动处理日期列
        data = self._handle_datetime_columns(data, categorical_columns=categorical_columns)

        if columns is None:
            columns = self._get_valid_numeric_columns(data, categorical_columns=categorical_columns)
        else:
            # 确保只处理数值列
            columns = self._get_valid_numeric_columns(data, columns, categorical_columns=categorical_columns)
        
        if method == 'isolation_forest':
            # 孤立森林是多变量方法，需要同时考虑所有列
            # 这里我们需要处理缺失值，因为IsolationForest不支持NaN
            X = data[columns].fillna(data[columns].mean())
            
            # contamination 是异常值比例的估计，我们可以基于threshold调整，或者固定一个值
            # 这里简单处理：threshold如果是小数且小于0.5，作为contamination
            # 否则使用默认 'auto'
            contamination = 'auto'
            if 0 < threshold < 0.5:
                contamination = threshold
                
            clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            y_pred = clf.fit_predict(X)
            
            # -1 表示异常值，1 表示正常值
            outliers_mask = y_pred == -1
            
            # 孤立森林是针对整体行的，但为了兼容现有结构，我们在每列都标记这一行为异常
            for col in columns:
                outliers_info['outliers_by_column'][col] = {
                    'count': int(outliers_mask.sum()),
                    'percentage': float((outliers_mask.sum() / len(data)) * 100),
                    'indices': data[outliers_mask].index.tolist(),
                    'lower_bound': None, # 孤立森林没有简单的上下界
                    'upper_bound': None
                }
                
        else:
            for col in columns:
                if method == 'iqr':
                    # IQR方法
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    
                elif method == 'zscore':
                    # Z-score方法
                    mean = data[col].mean()
                    std = data[col].std()
                    z_scores = np.abs((data[col] - mean) / std)
                    
                    outliers_mask = z_scores > threshold
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                else:
                    raise ValueError(f"未知的异常值检测方法: {method}")
                
                # 记录异常值信息
                outliers_info['outliers_by_column'][col] = {
                    'count': int(outliers_mask.sum()),
                    'percentage': float((outliers_mask.sum() / len(data)) * 100),
                    'indices': data[outliers_mask].index.tolist(),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        return data, outliers_info
    
    def handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'iqr',
        action: str = 'clip',
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理异常值
        
        Args:
            data: 数据集
            method: 检测方法 ('iqr', 'zscore')
            action: 处理方式 ('clip', 'remove', 'replace_mean', 'replace_median')
            columns: 要处理的列
            threshold: 阈值
            categorical_columns: 分类变量列名列表
            
        Returns:
            处理后的数据和处理信息
        """
        df = data.copy()
        
        # 检测异常值
        _, outliers_info = self.detect_outliers(df, method, columns, threshold, categorical_columns)
        
        info = {
            'method': method,
            'action': action,
            'threshold': threshold,
            'columns_processed': [],
            'outliers_before': {},
            'outliers_after': {}
        }
        
        if columns is None:
            columns = self._get_valid_numeric_columns(df, categorical_columns=categorical_columns)
        else:
            columns = self._get_valid_numeric_columns(df, columns, categorical_columns=categorical_columns)
        
        for col in columns:
            if col not in outliers_info['outliers_by_column']: continue
            col_info = outliers_info['outliers_by_column'][col]
            info['outliers_before'][col] = col_info['count']
            
            if col_info['count'] > 0:
                lower_bound = col_info['lower_bound']
                upper_bound = col_info['upper_bound']
                
                if action == 'clip':
                    # 将异常值裁剪到边界
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif action == 'remove':
                    # 删除包含异常值的行
                    mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                    df = df[mask]
                
                elif action == 'replace_mean':
                    # 用均值替换
                    mean_val = df[col].mean()
                    df[col] = df[col].apply(
                        lambda x: mean_val if x < lower_bound or x > upper_bound else x
                    )
                
                elif action == 'replace_median':
                    # 用中位数替换
                    median_val = df[col].median()
                    df[col] = df[col].apply(
                        lambda x: median_val if x < lower_bound or x > upper_bound else x
                    )
                
                info['columns_processed'].append(col)
        
        # 重新检测异常值
        _, outliers_after = self.detect_outliers(df, method, columns, threshold)
        for col in columns:
            if col in df.columns:
                info['outliers_after'][col] = outliers_after['outliers_by_column'][col]['count']
        
        self.preprocessing_history.append({
            'operation': 'handle_outliers',
            'info': info
        })
        
        return df, info
    
    # ==================== 特征工程 ====================
    
    def create_polynomial_features(
        self,
        data: pd.DataFrame,
        degree: int = 2,
        columns: Optional[List[str]] = None,
        interaction_only: bool = False,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        创建多项式特征
        
        Args:
            data: 数据集
            degree: 多项式次数
            columns: 要处理的列
            interaction_only: 是否仅创建交互特征
            categorical_columns: 分类变量列名列表
            
        Returns:
            新特征数据和信息
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        df = data.copy()
        
        # 自动处理日期列
        df = self._handle_datetime_columns(df, categorical_columns=categorical_columns)

        if columns is None:
            columns = self._get_valid_numeric_columns(df, categorical_columns=categorical_columns)
        else:
            columns = self._get_valid_numeric_columns(df, columns, categorical_columns=categorical_columns)
        
        # 创建多项式特征
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        poly_features = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)
        
        # 创建新的DataFrame
        poly_df = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=df.index
        )
        
        # 合并原始特征和多项式特征
        result_df = pd.concat([df, poly_df], axis=1)
        
        info = {
            'degree': degree,
            'interaction_only': interaction_only,
            'original_columns': columns,
            'new_columns': feature_names.tolist(),
            'n_new_features': len(feature_names)
        }
        
        self.preprocessing_history.append({
            'operation': 'create_polynomial_features',
            'info': info
        })
        
        return result_df, info

    def select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        method: str = 'variance',
        columns: Optional[List[str]] = None,
        k: int = 10,
        threshold: float = 0.0,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        特征选择
        
        Args:
            data: 数据集
            target_column: 目标变量
            method: 选择方法 ('variance', 'kbest', 'model')
            columns: 要选择的列
            k: 选择的特征数量 (method='kbest' or 'model')
            threshold: 阈值 (method='variance' or 'model')
            categorical_columns: 分类变量列名列表
            
        Returns:
            (处理后的数据, 信息)
        """
        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        df = data.copy()
        
        # 自动处理日期列
        df = self._handle_datetime_columns(df, categorical_columns=categorical_columns)
        
        # 确定特征列
        if columns is None:
            columns = [c for c in df.columns if c != target_column]
            
        # 筛选数值列
        numeric_columns = self._get_valid_numeric_columns(df, columns, categorical_columns=categorical_columns)
        
        # 如果有目标列，将其从特征中排除，但在最后保留
        target_data = None
        if target_column and target_column in df.columns:
            target_data = df[target_column]
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
        
        # 再次检查每一列，尝试转换为float，如果失败则排除（终极防御）
        # 针对 "Could not convert string ... to numeric" 的顽固错误
        safe_numeric_columns = []
        for col in numeric_columns:
            try:
                # 尝试取前10个非空值转换为float
                sample = df[col].dropna().head(10)
                if not sample.empty:
                    sample.astype(float)
                safe_numeric_columns.append(col)
            except Exception as e:
                logger.warning(f"列 {col} 看起来是数值型但在转换时失败，已从特征选择中排除: {e}")
        
        numeric_columns = safe_numeric_columns
        X = df[numeric_columns]
        selected_columns = numeric_columns
        dropped_columns = []
        feature_scores = {}
        
        if method == 'variance':
            # 方差选择法
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)
            selected_mask = selector.get_support()
            selected_columns = [col for i, col in enumerate(numeric_columns) if selected_mask[i]]
            dropped_columns = [col for i, col in enumerate(numeric_columns) if not selected_mask[i]]
            feature_scores = {col: float(var) for col, var in zip(numeric_columns, selector.variances_)}
            
        elif method == 'correlation':
            # 相关系数法 (移除高相关特征)
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            dropped_columns = to_drop
            selected_columns = [col for col in numeric_columns if col not in to_drop]
            
        elif method == 'importance':
            # 基于树模型的特征重要性
            if target_data is None:
                raise ValueError("基于重要性的特征选择需要目标列")
                
            # 简单的缺失值处理 (RandomForest不支持NaN)
            X_filled = X.fillna(X.mean())
            
            # 判断是回归还是分类
            # 简单判断: 如果目标变量唯一值少于10个或为object类型，则认为是分类
            is_classification = target_data.dtype == 'object' or target_data.nunique() < 10
            
            if is_classification:
                # 如果是分类，确保目标变量是数值编码
                if target_data.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(target_data)
                else:
                    y = target_data
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                y = target_data
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                
            model.fit(X_filled, y)
            
            importances = model.feature_importances_
            feature_scores = {col: float(imp) for col, imp in zip(numeric_columns, importances)}
            
            # 根据n_features或threshold选择
            if n_features:
                indices = np.argsort(importances)[::-1][:n_features]
                selected_columns = [numeric_columns[i] for i in indices]
            elif threshold > 0:
                selector = SelectFromModel(model, threshold=threshold, prefit=True)
                selected_mask = selector.get_support()
                selected_columns = [col for i, col in enumerate(numeric_columns) if selected_mask[i]]
            else:
                # 默认选择mean以上
                selector = SelectFromModel(model, threshold='mean', prefit=True)
                selected_mask = selector.get_support()
                selected_columns = [col for i, col in enumerate(numeric_columns) if selected_mask[i]]
                
            dropped_columns = [col for col in numeric_columns if col not in selected_columns]
            
        elif method == 'recursive':
            # 递归特征消除 (RFE)
            if target_data is None:
                raise ValueError("递归特征消除需要目标列")
                
            # 简单的缺失值处理
            X_filled = X.fillna(X.mean())
            
            # 判断是回归还是分类
            is_classification = target_data.dtype == 'object' or target_data.nunique() < 10
            
            if is_classification:
                if target_data.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(target_data)
                else:
                    y = target_data
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                y = target_data
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                
            n_features_to_select = n_features if n_features else (len(numeric_columns) // 2)
            rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
            rfe.fit(X_filled, y)
            
            selected_mask = rfe.support_
            selected_columns = [col for i, col in enumerate(numeric_columns) if selected_mask[i]]
            dropped_columns = [col for i, col in enumerate(numeric_columns) if not selected_mask[i]]
            feature_scores = {col: int(rank) for col, rank in zip(numeric_columns, rfe.ranking_)}
            
        else:
            raise ValueError(f"未知的特征选择方法: {method}")
        
        # 构建结果DataFrame
        # 保留未参与选择的非数值列
        other_columns = [col for col in df.columns if col not in numeric_columns and col != target_column]
        
        result_columns = selected_columns + other_columns
        if target_column and target_data is not None:
            result_columns.append(target_column)
            
        result_df = df[result_columns]
        
        info = {
            'method': method,
            'original_n_features': len(numeric_columns),
            'selected_n_features': len(selected_columns),
            'dropped_n_features': len(dropped_columns),
            'selected_columns': selected_columns,
            'dropped_columns': dropped_columns,
            'feature_scores': feature_scores
        }
        
        self.preprocessing_history.append({
            'operation': 'select_features',
            'info': info
        })
        
        return result_df, info
    
    # ==================== 工具方法 ====================
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """获取预处理历史摘要"""
        return {
            'total_operations': len(self.preprocessing_history),
            'operations': self.preprocessing_history,
            'scalers_saved': list(self.scalers.keys()),
            'imputers_saved': list(self.imputers.keys())
        }
    
    def reset(self):
        """重置预处理服务"""
        self.scalers = {}
        self.imputers = {}
        self.preprocessing_history = []
