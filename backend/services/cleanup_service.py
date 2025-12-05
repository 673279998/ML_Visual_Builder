import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from backend.database.db_manager import DatabaseManager
from backend.services.data_service import DataService
from backend.config import DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, PREPROCESSOR_DIR, ENCODER_DIR

logger = logging.getLogger(__name__)

class CleanupService:
    def __init__(self, db_manager: DatabaseManager, data_service: DataService):
        self.db = db_manager
        self.data_service = data_service

    def _normalize_path(self, path: str) -> str:
        """标准化路径用于比较"""
        return str(Path(path).resolve())

    def cleanup_all(self) -> Dict[str, int]:
        """执行所有清理任务"""
        results = {
            'datasets_deleted': 0,
            'models_deleted': 0,
            'predictions_deleted': 0,
            'components_deleted': 0,
            'prediction_records_deleted': 0
        }
        
        try:
            results['datasets_deleted'] = self.cleanup_orphaned_processed_datasets()
            results['models_deleted'] = self.cleanup_orphaned_models()
            
            pred_res = self.cleanup_orphaned_predictions()
            results['predictions_deleted'] = pred_res['files']
            results['prediction_records_deleted'] = pred_res['records']
            
            results['components_deleted'] = self.cleanup_orphaned_components()
        except Exception as e:
            logger.error(f"清理任务失败: {e}")
            raise e
            
        return results

    def cleanup_orphaned_processed_datasets(self) -> int:
        """清理无关联的处理后数据集"""
        count = 0
        # 获取所有处理后的数据集 (source_dataset_id 不为空)
        query = "SELECT id, name FROM datasets WHERE source_dataset_id IS NOT NULL"
        datasets = self.db.execute_query(query)
        
        for ds in datasets:
            ds_id = ds['id']
            # 检查是否有模型使用
            model_usage = self.db.execute_query("SELECT count(*) as c FROM models WHERE dataset_id = ?", (ds_id,))
            if model_usage and model_usage[0]['c'] > 0:
                continue
                
            # 检查是否有预测任务使用
            pred_usage = self.db.execute_query("SELECT count(*) as c FROM predictions WHERE input_dataset_id = ?", (ds_id,))
            if pred_usage and pred_usage[0]['c'] > 0:
                continue
            
            # 检查是否是其他数据集的源 (虽然处理后的数据集一般不会再作为源，但为了安全)
            source_usage = self.db.execute_query("SELECT count(*) as c FROM datasets WHERE source_dataset_id = ?", (ds_id,))
            if source_usage and source_usage[0]['c'] > 0:
                continue
                
            # 如果未被使用，则删除
            logger.info(f"删除无关联的处理后数据集: {ds['name']} (ID: {ds_id})")
            try:
                if self.data_service.delete_dataset(ds_id):
                    count += 1
            except Exception as e:
                logger.error(f"删除数据集失败 {ds_id}: {e}")
                
        return count

    def cleanup_orphaned_models(self) -> int:
        """清理无关联的模型文件"""
        count = 0
        # 获取数据库中有效的模型文件路径和文件名
        valid_files = set()
        valid_filenames = set()
        
        models = self.db.execute_query("SELECT model_file_path FROM models")
        for m in models:
            if m['model_file_path']:
                try:
                    path_obj = Path(m['model_file_path'])
                    valid_files.add(self._normalize_path(m['model_file_path']))
                    valid_filenames.add(path_obj.name)
                except Exception:
                    pass
        
        # 扫描模型目录
        import time
        current_time = time.time()
        # 设置保护期为1小时 (3600秒)
        # 只有修改时间超过1小时且不在数据库中的文件才会被删除
        PROTECTION_PERIOD = 3600
        
        if MODELS_DIR.exists():
            for file_path in MODELS_DIR.glob('**/*'):
                if file_path.is_file():
                    try:
                        # 1. 检查是否在保护期内
                        mtime = os.path.getmtime(file_path)
                        if current_time - mtime < PROTECTION_PERIOD:
                            continue
                            
                        # 2. 检查文件名是否匹配 (作为第二道防线，防止路径解析差异)
                        if file_path.name in valid_filenames:
                            continue
                            
                        # 3. 检查全路径
                        normalized = self._normalize_path(str(file_path))
                        if normalized not in valid_files:
                            os.remove(file_path)
                            logger.info(f"已删除孤立的模型文件: {file_path}")
                            count += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
        return count

    def cleanup_orphaned_predictions(self) -> Dict[str, int]:
        """清理无关联的预测结果"""
        files_deleted = 0
        records_deleted = 0
        
        # 1. 清理数据库中关联了不存在模型的预测记录
        query = """
            SELECT p.id, p.output_file_path 
            FROM predictions p 
            LEFT JOIN models m ON p.model_id = m.id 
            WHERE m.id IS NULL
        """
        orphaned_records = self.db.execute_query(query)
        for record in orphaned_records:
            # 删除文件
            if record['output_file_path'] and os.path.exists(record['output_file_path']):
                try:
                    os.remove(record['output_file_path'])
                    files_deleted += 1
                except Exception as e:
                    logger.error(f"删除预测文件失败 {record['output_file_path']}: {e}")
            
            # 删除记录
            try:
                self.db.execute_update("DELETE FROM predictions WHERE id = ?", (record['id'],))
                records_deleted += 1
            except Exception as e:
                logger.error(f"删除预测记录失败 {record['id']}: {e}")
            
        # 2. 清理文件系统中未在数据库记录的文件
        valid_files = set()
        predictions = self.db.execute_query("SELECT output_file_path FROM predictions")
        for p in predictions:
            if p['output_file_path']:
                try:
                    valid_files.add(self._normalize_path(p['output_file_path']))
                except Exception:
                    pass
                
        if PREDICTIONS_DIR.exists():
            for file_path in PREDICTIONS_DIR.glob('**/*'):
                if file_path.is_file():
                    try:
                        normalized = self._normalize_path(str(file_path))
                        if normalized not in valid_files:
                            os.remove(file_path)
                            logger.info(f"已删除孤立的预测文件: {file_path}")
                            files_deleted += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
                            
        return {'files': files_deleted, 'records': records_deleted}

    def cleanup_orphaned_components(self) -> int:
        """清理无关联的编码器和预处理器"""
        count = 0
        valid_files = set()
        valid_filenames = set() # 新增文件名集合
        
        # 1. 获取 preprocessing_components 表中的文件路径
        components = self.db.execute_query("SELECT file_path FROM preprocessing_components")
        for c in components:
            if c['file_path']:
                try:
                    path_obj = Path(c['file_path'])
                    valid_files.add(self._normalize_path(c['file_path']))
                    valid_filenames.add(path_obj.name)
                except Exception:
                    pass
                    
        # 2. 获取 dataset_encoders 表中的文件路径
        encoders = self.db.execute_query("SELECT file_path FROM dataset_encoders")
        for e in encoders:
            if e['file_path']:
                try:
                    path_obj = Path(e['file_path'])
                    valid_files.add(self._normalize_path(e['file_path']))
                    valid_filenames.add(path_obj.name)
                except Exception:
                    pass
        
        # 3. 获取所有有效的模型ID，用于保护模型组件目录
        active_model_ids = set()
        models = self.db.execute_query("SELECT id FROM models")
        for m in models:
            active_model_ids.add(str(m['id']))
        # 额外：保护被活跃模型使用的数据集ID，对其 encoders/preprocessors 目录整体跳过
        protected_dataset_ids = set()
        model_datasets = self.db.execute_query("SELECT dataset_id FROM models WHERE dataset_id IS NOT NULL")
        for md in model_datasets:
            protected_dataset_ids.add(str(md['dataset_id']))
            
        import time
        current_time = time.time()
        PROTECTION_PERIOD = 3600 # 1小时保护期
        
        # 处理公共组件目录 (dataset_encoders, dataset_preprocessors)
        for base_dir in [ENCODER_DIR, PREPROCESSOR_DIR]:
            if base_dir.exists():
                for file_path in base_dir.glob('**/*.pkl'):
                    if file_path.is_file():
                        try:
                            # 1. 检查是否在保护期内
                            mtime = os.path.getmtime(file_path)
                            if current_time - mtime < PROTECTION_PERIOD:
                                continue
                                
                            # 2. 检查文件名
                            if file_path.name in valid_filenames:
                                continue

                            # 如果文件位于活跃模型使用的数据集目录下，则整体保护
                            try:
                                rel = file_path.relative_to(base_dir)
                                dataset_dir = rel.parts[0] if len(rel.parts) > 0 else ''
                                if dataset_dir and dataset_dir in protected_dataset_ids:
                                    continue
                            except Exception:
                                pass
                            normalized = self._normalize_path(str(file_path))
                            if normalized not in valid_files:
                                os.remove(file_path)
                                logger.info(f"已删除孤立的组件文件: {file_path}")
                                count += 1
                        except Exception as e:
                            logger.error(f"删除组件失败 {file_path}: {e}")
                                
                # 清理空目录
                for dir_path in base_dir.glob('*'):
                    if dir_path.is_dir() and not any(dir_path.iterdir()):
                        try:
                            dir_path.rmdir()
                            logger.info(f"已删除空目录: {dir_path}")
                        except Exception:
                            pass
        return count
