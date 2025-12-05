"""
数据处理服务
"""
import os
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from werkzeug.utils import secure_filename
from backend.config import UPLOAD_DIR, DATA_DIR
from backend.database.db_manager import DatabaseManager


class DataService:
    """数据处理服务类"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def upload_and_parse_file(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        上传并解析数据文件
        
        Args:
            file_path: 文件路径
            original_filename: 原始文件名
            
        Returns:
            包含数据集信息的字典
        """
        # 获取文件格式
        file_format = self._get_file_format(original_filename)
        
        # 读取文件
        df = self._read_file(file_path, file_format)
        
        # 推断数据类型
        column_info = self._infer_data_types(df)
        
        # 生成统计信息
        statistics = self._generate_statistics(df, column_info)
        
        # 保存到数据库
        dataset_id = self._save_to_database(
            name=original_filename,
            file_path=file_path,
            file_format=file_format,
            df=df,
            column_info=column_info,
            statistics=statistics
        )
        
        return {
            'dataset_id': dataset_id,
            'name': original_filename,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': column_info,
            'preview': df.head(10).to_dict('records')
        }
    
    def _get_file_format(self, filename: str) -> str:
        """获取文件格式"""
        ext = filename.lower().split('.')[-1]
        format_map = {
            'csv': 'csv',
            'xlsx': 'excel',
            'xls': 'excel',
            'json': 'json'
        }
        return format_map.get(ext, 'unknown')
    
    def _read_file(self, file_path: str, file_format: str) -> pd.DataFrame:
        """读取文件"""
        if file_format == 'csv':
            return pd.read_csv(file_path)
        elif file_format == 'excel':
            return pd.read_excel(file_path)
        elif file_format == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
    
    def _infer_data_types(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        推断数据类型
        
        Returns:
            列信息列表
        """
        column_info = []
        
        for idx, col in enumerate(df.columns):
            col_data = df[col]
            data_type = self._infer_column_type(col_data)
            missing_count = col_data.isna().sum()
            unique_count = col_data.nunique()
            
            column_info.append({
                'column_name': col,
                'column_index': idx,
                'data_type': data_type,
                'missing_count': int(missing_count),
                'unique_count': int(unique_count)
            })
        
        return column_info
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """
        推断单列的数据类型
        
        Returns:
            'numeric', 'categorical', 'datetime'
        """
        # 排除缺失值
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return 'categorical'
        
        # 尝试转换为数值型
        try:
            pd.to_numeric(non_null)
            return 'numeric'
        except (ValueError, TypeError):
            pass
        
        # 尝试转换为日期型
        try:
            # 使用 errors='raise' 以便非日期数据触发异常，避免 pandas 发出警告
            pd.to_datetime(non_null, errors='raise')
            # 检查是否看起来像日期
            sample = non_null.iloc[0] if len(non_null) > 0 else None
            if sample and isinstance(sample, str):
                # 简单的日期格式检查
                if any(sep in str(sample) for sep in ['-', '/', ':']):
                    return 'datetime'
        except (ValueError, TypeError):
            pass
        
        # 默认为分类型
        return 'categorical'
    
    def _generate_statistics(self, df: pd.DataFrame, column_info: List[Dict]) -> Dict[str, Dict]:
        """生成统计信息"""
        from backend.utils.json_utils import sanitize_for_json
        statistics = {}
        
        for col_info in column_info:
            col_name = col_info['column_name']
            data_type = col_info['data_type']
            col_data = df[col_name]
            
            if data_type == 'numeric':
                # 确保数据是数值型，避免对看起来像数字的字符串列进行操作导致报错 (如 '0.0' + '0.0' -> '0.00.0')
                if col_data.dtype == 'object':
                    col_data = pd.to_numeric(col_data, errors='coerce')
                    
                stats = {
                    'mean': float(col_data.mean()) if not col_data.isna().all() else None,
                    'median': float(col_data.median()) if not col_data.isna().all() else None,
                    'std': float(col_data.std()) if not col_data.isna().all() else None,
                    'min': float(col_data.min()) if not col_data.isna().all() else None,
                    'max': float(col_data.max()) if not col_data.isna().all() else None,
                    'q25': float(col_data.quantile(0.25)) if not col_data.isna().all() else None,
                    'q75': float(col_data.quantile(0.75)) if not col_data.isna().all() else None
                }
            elif data_type == 'categorical':
                value_counts = col_data.value_counts().head(10).to_dict()
                # 确保unique_values中的值都是Python原生类型，不是numpy类型
                unique_vals = col_data.unique()[:50]
                # 使用sanitize_for_json处理numpy类型
                safe_unique_vals = sanitize_for_json(unique_vals.tolist() if hasattr(unique_vals, 'tolist') else list(unique_vals))
                
                stats = {
                    'unique_values': safe_unique_vals,
                    'value_counts': {str(k): int(v) for k, v in value_counts.items()},
                    'mode': str(col_data.mode()[0]) if not col_data.mode().empty else None
                }
            elif data_type == 'datetime':
                try:
                    dt_col = pd.to_datetime(col_data, errors='coerce')
                    stats = {
                        'min_date': str(dt_col.min()) if not dt_col.isna().all() else None,
                        'max_date': str(dt_col.max()) if not dt_col.isna().all() else None,
                        'date_range_days': int((dt_col.max() - dt_col.min()).days) if not dt_col.isna().all() else None
                    }
                except:
                    stats = {}
            else:
                stats = {}
            
            # 使用sanitize_for_json确保统计信息中没有numpy类型
            statistics[col_name] = sanitize_for_json(stats)
        
        return statistics
    
    def _save_to_database(self, name: str, file_path: str, file_format: str,
                          df: pd.DataFrame, column_info: List[Dict],
                          statistics: Dict[str, Dict],
                          source_dataset_id: Optional[int] = None) -> int:
        """保存数据到数据库"""
        # 创建数据集记录
        dataset_id = self.db.create_dataset(
            name=name,
            file_path=file_path,
            file_format=file_format,
            row_count=len(df),
            column_count=len(df.columns),
            source_dataset_id=source_dataset_id
        )
        
        # 创建列记录
        for col_info in column_info:
            col_name = col_info['column_name']
            self.db.create_column(
                dataset_id=dataset_id,
                column_name=col_name,
                column_index=col_info.get('column_index', 0),
                data_type=col_info['data_type'],
                is_target=col_info.get('is_target', False),
                is_excluded=col_info.get('is_excluded', False),
                encoding_method=col_info.get('encoding_method'),
                encoding_config=col_info.get('encoding_config'),
                missing_count=col_info.get('missing_count', 0),
                unique_count=col_info.get('unique_count'),
                statistics=statistics.get(col_name, {})
            )
        
        return dataset_id
    
    def save_dataset(self, data: pd.DataFrame, name: str, source_dataset_id: int) -> int:
        """
        保存处理后的数据集
        
        Args:
            data: 处理后的DataFrame
            name: 新数据集名称
            source_dataset_id: 源数据集ID
            
        Returns:
            新数据集ID
        """
        # 生成安全的文件名
        safe_name = secure_filename(name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_{timestamp}.csv"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # 保存为CSV
        data.to_csv(file_path, index=False)
        
        # 推断数据类型和统计信息
        column_info = self._infer_data_types(data)
        
        # 如果有源数据集，尝试继承列配置 (is_target, is_excluded)
        if source_dataset_id:
            source_columns = self.db.get_columns(source_dataset_id)
            source_col_map = {col['column_name']: col for col in source_columns}
            
            for col in column_info:
                col_name = col['column_name']
                if col_name in source_col_map:
                    source_col = source_col_map[col_name]
                    col['is_target'] = source_col.get('is_target', False)
                    col['is_excluded'] = source_col.get('is_excluded', False)
                    col['encoding_method'] = source_col.get('encoding_method')
                    col['encoding_config'] = source_col.get('encoding_config')
                    
                    # 关键修复：如果源数据中定义为categorical，强制保留该类型，避免被重新推断为numeric
                    # 这样可以避免对分类变量计算mean/std等数值统计量
                    if source_col.get('data_type') == 'categorical':
                        col['data_type'] = 'categorical'
        
        statistics = self._generate_statistics(data, column_info)
        
        # 保存到数据库
        return self._save_to_database(
            name=name,
            file_path=file_path,
            file_format='csv',
            df=data,
            column_info=column_info,
            statistics=statistics,
            source_dataset_id=source_dataset_id
        )

    def load_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """
        加载完整数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            包含数据集信息和DataFrame的字典
        """
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        # 读取数据文件
        df = self._read_file(dataset['file_path'], dataset['file_format'])
        
        # 获取列配置并强制应用数据类型
        # 解决"分类变量被误读为连续数值"的问题
        columns_config = self.db.get_columns(dataset_id)
        if columns_config:
            for col_config in columns_config:
                col_name = col_config['column_name']
                data_type = col_config['data_type']
                
                if col_name in df.columns:
                    if data_type == 'categorical':
                        # 强制转换为字符串(object)
                        df[col_name] = df[col_name].astype(str)
                    elif data_type == 'numeric':
                        # 尝试转换为数值
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        return {
            'dataset': dataset,
            'data': df
        }

    def get_dataset_data(self, dataset_id: int, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """获取数据集内容（分页）"""
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        # 读取数据文件
        df = self._read_file(dataset['file_path'], dataset['file_format'])
        
        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = df.iloc[start_idx:end_idx]
        
        # 替换 NaN 为 None，以便 JSON 序列化
        page_data = page_data.replace({np.nan: None})
        
        return {
            'total_rows': len(df),
            'page': page,
            'page_size': page_size,
            'data': page_data.to_dict('records')
        }
    
    def update_column_properties(self, column_id: int, **properties) -> None:
        """更新列属性"""
        self.db.update_column(column_id, **properties)
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """
        删除数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            是否成功删除
        """
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            return False
            
        # 删除文件
        try:
            if os.path.exists(dataset['file_path']):
                os.remove(dataset['file_path'])
        except OSError:
            pass  # 文件可能已经被删除了，忽略错误
            
        # 删除关联的encoders和preprocessors目录
        # 注意：不应在此处删除组件目录，因为这些组件可能被已训练的模型所引用。
        # 清理工作应由 cleanup_service 在确认无引用后进行。
        # encoder_dir = DATA_DIR / 'encoders' / str(dataset_id)
        # preprocessor_dir = DATA_DIR / 'preprocessors' / str(dataset_id)
        
        # for dir_path in [encoder_dir, preprocessor_dir]:
        #     if dir_path.exists() and dir_path.is_dir():
        #         try:
        #             shutil.rmtree(dir_path)
        #             print(f"已删除目录: {dir_path}")
        #         except OSError as e:
        #             print(f"删除目录失败 {dir_path}: {e}")
            
        # 删除数据库记录
        return self.db.delete_dataset(dataset_id)
    
    def batch_delete_datasets(self, dataset_ids: List[int]) -> Dict[str, Any]:
        """
        批量删除数据集
        
        Args:
            dataset_ids: 数据集ID列表
            
        Returns:
            删除结果摘要
        """
        results = {
            'success': [],
            'failed': []
        }
        
        for dataset_id in dataset_ids:
            try:
                if self.delete_dataset(dataset_id):
                    results['success'].append(dataset_id)
                else:
                    results['failed'].append({'id': dataset_id, 'error': 'Dataset not found'})
            except Exception as e:
                results['failed'].append({'id': dataset_id, 'error': str(e)})
                
        return results

    def save_dataset_data(self, dataset_id: int, data: List[Dict[str, Any]]) -> bool:
        """
        保存数据集数据
        
        Args:
            dataset_id: 数据集ID
            data: 数据列表（字典列表）
            
        Returns:
            是否成功保存
        """
        print(f"Saving dataset {dataset_id} with {len(data)} rows")
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
            
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 保存文件
        file_path = dataset['file_path']
        file_format = dataset['file_format']
        print(f"Writing to file: {file_path} (format: {file_format})")
        
        try:
            if file_format == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format == 'excel':
                df.to_excel(file_path, index=False)
            elif file_format == 'json':
                df.to_json(file_path, orient='records')
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")
                
            print(f"Successfully wrote file for dataset {dataset_id}")
            
            # 更新数据库中的统计信息（可选，这里简化处理，只更新行数）
            # 实际上应该重新计算统计信息并更新列信息，但这比较耗时
            self.db.update_dataset(dataset_id, row_count=len(df))
            
            return True
        except Exception as e:
            print(f"Error saving dataset {dataset_id}: {str(e)}")
            raise e

    def get_dataset_info(self, dataset_id: int) -> Dict[str, Any]:
        """获取数据集完整信息"""
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        columns = self.db.get_columns(dataset_id)
        
        return {
            'dataset': dataset,
            'columns': columns
        }
