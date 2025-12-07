"""
数据库管理器
"""
import sqlite3
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import DATABASE_PATH, DATA_DIR


class DatabaseManager:
    """数据库管理器类"""
    
    def __init__(self):
        self.db_path = str(DATABASE_PATH)
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 返回字典格式的行
        return conn
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """执行查询并返回结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """执行插入操作并返回新记录的ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        lastrowid = cursor.lastrowid
        conn.close()
        return lastrowid

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """执行更新/删除操作并返回影响的行数"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        rowcount = cursor.rowcount
        conn.close()
        return rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """执行批量操作"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
        conn.close()
    
    # Datasets相关方法
    def create_dataset(self, name: str, file_path: str, file_format: str, 
                      row_count: int, column_count: int, 
                      is_encoded: bool = False, 
                      source_dataset_id: Optional[int] = None,
                      encoder_id: Optional[int] = None) -> int:
        """创建数据集记录"""
        query = '''
            INSERT INTO datasets (name, file_path, file_format, row_count, column_count, 
                                 is_encoded, source_dataset_id, encoder_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        return self.execute_insert(query, (name, file_path, file_format, row_count, 
                                          column_count, is_encoded, source_dataset_id, encoder_id))
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """获取数据集详情"""
        query = 'SELECT * FROM datasets WHERE id = ?'
        results = self.execute_query(query, (dataset_id,))
        return results[0] if results else None
    
    def get_all_datasets(self) -> List[Dict]:
        """获取所有数据集"""
        query = 'SELECT * FROM datasets ORDER BY created_at DESC'
        return self.execute_query(query)
    
    def update_dataset(self, dataset_id: int, **kwargs) -> None:
        """更新数据集"""
        set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
        query = f'UPDATE datasets SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?'
        values = list(kwargs.values()) + [dataset_id]
        self.execute_update(query, tuple(values))
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """删除数据集"""
        # 检查是否有模型正在使用该数据集
        models = self.execute_query('SELECT id, name FROM models WHERE dataset_id = ?', (dataset_id,))
        if models:
            model_names = ", ".join([m['name'] for m in models])
            raise ValueError(f"无法删除数据集，因为它被以下模型使用: {model_names}")

        # 获取数据集信息以删除文件
        dataset = self.get_dataset(dataset_id)
        if dataset and dataset.get('file_path') and os.path.exists(dataset['file_path']):
            try:
                os.remove(dataset['file_path'])
            except OSError as e:
                print(f"删除数据集文件失败: {e}")

        # 先删除关联的列信息
        self.execute_update('DELETE FROM dataset_columns WHERE dataset_id = ?', (dataset_id,))
        # 删除关联的编码器记录 (如果有)
        self.execute_update('DELETE FROM dataset_encoders WHERE encoded_dataset_id = ?', (dataset_id,))
        
        # 删除数据集
        rows_affected = self.execute_update('DELETE FROM datasets WHERE id = ?', (dataset_id,))
        return rows_affected > 0
    
    # Dataset Columns相关方法
    def create_column(self, dataset_id: int, column_name: str, column_index: int,
                     data_type: str, is_target: bool = False, is_excluded: bool = False,
                     encoding_method: Optional[str] = None, encoding_config: Optional[Dict] = None,
                     missing_count: int = 0, unique_count: Optional[int] = None,
                     statistics: Optional[Dict] = None) -> int:
        """创建列记录"""
        query = '''
            INSERT INTO dataset_columns (dataset_id, column_name, column_index, data_type,
                                        is_target, is_excluded, encoding_method, encoding_config,
                                        missing_count, unique_count, statistics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        encoding_config_json = json.dumps(encoding_config) if encoding_config else None
        statistics_json = json.dumps(statistics) if statistics else None
        return self.execute_insert(query, (dataset_id, column_name, column_index, data_type,
                                          is_target, is_excluded, encoding_method, encoding_config_json,
                                          missing_count, unique_count, statistics_json))
    
    def get_columns(self, dataset_id: int) -> List[Dict]:
        """获取数据集的所有列"""
        query = 'SELECT * FROM dataset_columns WHERE dataset_id = ? ORDER BY column_index'
        columns = self.execute_query(query, (dataset_id,))
        # 解析JSON字段
        for col in columns:
            if col['encoding_config']:
                col['encoding_config'] = json.loads(col['encoding_config'])
            if col['statistics']:
                col['statistics'] = json.loads(col['statistics'])
        return columns
    
    def update_column(self, column_id: int, **kwargs) -> None:
        """更新列信息"""
        # 处理JSON字段
        if 'encoding_config' in kwargs and kwargs['encoding_config'] is not None:
            kwargs['encoding_config'] = json.dumps(kwargs['encoding_config'])
        if 'statistics' in kwargs and kwargs['statistics'] is not None:
            kwargs['statistics'] = json.dumps(kwargs['statistics'])
        
        set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
        query = f'UPDATE dataset_columns SET {set_clause} WHERE id = ?'
        values = list(kwargs.values()) + [column_id]
        self.execute_update(query, tuple(values))
    
    # Dataset Encoders相关方法
    def create_encoder(self, name: str, source_dataset_id: int, encoded_dataset_id: int,
                      file_path: str, column_mappings: Dict, encoding_summary: Dict,
                      workflow_id: Optional[str] = None) -> int:
        """创建编码器记录"""
        query = '''
            INSERT INTO dataset_encoders (name, source_dataset_id, encoded_dataset_id,
                                         file_path, column_mappings, encoding_summary, workflow_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        return self.execute_insert(query, (name, source_dataset_id, encoded_dataset_id,
                                          file_path, json.dumps(column_mappings), 
                                          json.dumps(encoding_summary), workflow_id))
    
    def get_encoder(self, encoder_id: int) -> Optional[Dict]:
        """获取编码器详情"""
        query = 'SELECT * FROM dataset_encoders WHERE id = ?'
        results = self.execute_query(query, (encoder_id,))
        if results:
            encoder = results[0]
            encoder['column_mappings'] = json.loads(encoder['column_mappings'])
            encoder['encoding_summary'] = json.loads(encoder['encoding_summary'])
            return encoder
        return None
    
    # Workflows相关方法
    def create_workflow(self, name: str, workflow_type: str, configuration: Dict,
                       description: Optional[str] = None) -> int:
        """创建工作流"""
        query = '''
            INSERT INTO workflows (name, workflow_type, description, configuration)
            VALUES (?, ?, ?, ?)
        '''
        return self.execute_insert(query, (name, workflow_type, description, 
                                          json.dumps(configuration)))
    
    def get_workflow(self, workflow_id: int) -> Optional[Dict]:
        """获取工作流详情"""
        query = 'SELECT * FROM workflows WHERE id = ?'
        results = self.execute_query(query, (workflow_id,))
        if results:
            workflow = results[0]
            workflow['configuration'] = json.loads(workflow['configuration'])
            return workflow
        return None
    
    def get_all_workflows(self) -> List[Dict]:
        """获取所有工作流"""
        query = 'SELECT * FROM workflows ORDER BY created_at DESC'
        workflows = self.execute_query(query)
        for wf in workflows:
            wf['configuration'] = json.loads(wf['configuration'])
        return workflows
    
    def update_workflow(self, workflow_id: int, **kwargs) -> None:
        """更新工作流"""
        if 'configuration' in kwargs:
            kwargs['configuration'] = json.dumps(kwargs['configuration'])
        
        set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
        query = f'UPDATE workflows SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?'
        values = list(kwargs.values()) + [workflow_id]
        self.execute_update(query, tuple(values))
    
    # Models相关方法
    def create_model(self, name: str, algorithm_type: str, algorithm_name: str,
                    dataset_id: int, model_file_path: str, hyperparameters: Dict,
                    performance_metrics: Dict, dataset_schema: Dict, input_requirements: Dict,
                    workflow_id: Optional[int] = None, encoder_id: Optional[int] = None,
                    feature_importance: Optional[Dict] = None,
                    actual_hyperparameters: Optional[Dict] = None,
                    complete_results: Optional[Dict] = None) -> int:
        """创建模型记录"""
        query = '''
            INSERT INTO models (name, algorithm_type, algorithm_name, workflow_id, dataset_id,
                               encoder_id, model_file_path, hyperparameters, performance_metrics,
                               dataset_schema, input_requirements, feature_importance,
                               actual_hyperparameters, complete_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        return self.execute_insert(query, (
            name, algorithm_type, algorithm_name, workflow_id, dataset_id, encoder_id,
            model_file_path, json.dumps(hyperparameters), json.dumps(performance_metrics),
            json.dumps(dataset_schema), json.dumps(input_requirements),
            json.dumps(feature_importance) if feature_importance else None,
            json.dumps(actual_hyperparameters) if actual_hyperparameters else None,
            json.dumps(complete_results) if complete_results else None
        ))
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        """获取模型详情"""
        query = 'SELECT * FROM models WHERE id = ?'
        results = self.execute_query(query, (model_id,))
        if results:
            model = results[0]
            model['hyperparameters'] = json.loads(model['hyperparameters'])
            model['performance_metrics'] = json.loads(model['performance_metrics'])
            model['dataset_schema'] = json.loads(model['dataset_schema'])
            model['input_requirements'] = json.loads(model['input_requirements'])
            if model['feature_importance']:
                model['feature_importance'] = json.loads(model['feature_importance'])
            if model.get('actual_hyperparameters'):
                model['actual_hyperparameters'] = json.loads(model['actual_hyperparameters'])
            if model.get('complete_results'):
                model['complete_results'] = json.loads(model['complete_results'])
            return model
        return None
    
    def get_all_models(self) -> List[Dict]:
        """获取所有模型"""
        query = 'SELECT * FROM models ORDER BY created_at DESC'
        models = self.execute_query(query)
        for model in models:
            model['hyperparameters'] = json.loads(model['hyperparameters'])
            model['performance_metrics'] = json.loads(model['performance_metrics'])
            model['dataset_schema'] = json.loads(model['dataset_schema'])
            model['input_requirements'] = json.loads(model['input_requirements'])
            if model['feature_importance']:
                model['feature_importance'] = json.loads(model['feature_importance'])
            if model.get('actual_hyperparameters'):
                model['actual_hyperparameters'] = json.loads(model['actual_hyperparameters'])
            if model.get('complete_results'):
                model['complete_results'] = json.loads(model['complete_results'])
        return models
    
    def delete_model(self, model_id: int) -> bool:
        """删除模型及其相关文件"""
        # 1. 获取模型信息
        model = self.get_model(model_id)
        if not model:
            return False
            
        dataset_id = model.get('dataset_id')
        workflow_id = model.get('workflow_id')
        
        # 删除模型文件
        if model.get('model_file_path') and os.path.exists(model['model_file_path']):
            try:
                os.remove(model['model_file_path'])
            except OSError as e:
                print(f"删除模型文件失败: {e}")
        
        # 尝试删除训练结果文件
        results_dir = DATA_DIR / 'training_results'
        result_filename = f"results_{model_id}_{model['algorithm_name']}.json"
        result_path = results_dir / result_filename
        if result_path.exists():
            try:
                os.remove(result_path)
            except OSError as e:
                print(f"删除训练结果文件失败: {e}")

        # 2. 检查是否需要删除关联的中间数据集
        # 如果模型使用的是处理后的数据集且不再被其他模型使用，则应该删除该数据集
        should_delete_dataset = False
        if dataset_id:
            ds = self.get_dataset(dataset_id)
            if ds and ds.get('source_dataset_id'):
                # 这是一个中间数据集
                # 检查是否被其他模型使用
                count = self.execute_query(
                    "SELECT COUNT(*) as c FROM models WHERE dataset_id = ? AND id != ?", 
                    (dataset_id, model_id)
                )[0]['c']
                
                if count == 0:
                    should_delete_dataset = True
                    # 关键步骤：如果决定删除数据集，先获取并删除其关联的 encoders 文件和记录
                    # 获取编码器文件
                    encoders = self.execute_query('SELECT file_path FROM dataset_encoders WHERE encoded_dataset_id = ?', (dataset_id,))
                    for enc in encoders:
                        f_path = enc['file_path']
                        if f_path and os.path.exists(f_path):
                            try:
                                os.remove(f_path)
                                print(f"已删除关联的编码器文件: {f_path}")
                                # 尝试删除父目录（如果为空）
                                parent_dir = os.path.dirname(f_path)
                                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                                    try:
                                        os.rmdir(parent_dir)
                                    except:
                                        pass
                            except OSError as e:
                                print(f"删除编码器文件失败: {e}")

                    self.execute_update('DELETE FROM dataset_encoders WHERE encoded_dataset_id = ?', (dataset_id,))

        # 3. 获取并删除预处理组件文件
        components = self.get_preprocessing_components(model_id)
        for comp in components:
            file_path = comp.get('file_path')
            # 处理相对路径
            if file_path and not os.path.isabs(file_path):
                from backend.config import BASE_DIR
                file_path = os.path.join(BASE_DIR, file_path)

            if file_path and os.path.exists(file_path):
                # 检查文件是否被其他模型使用
                other_model_usage = self.execute_query(
                    "SELECT COUNT(*) as c FROM preprocessing_components WHERE file_path = ? AND model_id != ?", 
                    (file_path, model_id)
                )[0]['c']
                
                # 检查文件是否被数据集记录使用
                # 如果上面执行了 delete from dataset_encoders，这里的 count 将为 0
                dataset_usage = self.execute_query(
                    "SELECT COUNT(*) as c FROM dataset_encoders WHERE file_path = ?", 
                    (file_path,)
                )[0]['c']
                
                if other_model_usage == 0 and dataset_usage == 0:
                    try:
                        os.remove(file_path)
                        print(f"已删除无关联的预处理组件文件: {file_path}")
                        # 尝试删除父目录（如果为空）
                        parent_dir = os.path.dirname(file_path)
                        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                            try:
                                os.rmdir(parent_dir)
                            except:
                                pass
                    except OSError as e:
                        print(f"删除预处理组件文件失败: {e}")
                else:
                    print(f"保留预处理组件文件(仍被使用 - 模型:{other_model_usage}, 数据集:{dataset_usage}): {file_path}")

        # 4. 获取并删除预测结果文件
        predictions = self.get_model_predictions(model_id)
        for pred in predictions:
            if pred.get('output_file_path') and os.path.exists(pred['output_file_path']):
                try:
                    os.remove(pred['output_file_path'])
                except OSError as e:
                    print(f"删除预测结果文件失败: {e}")

        # 5. 清理空目录 (Encoders/Preprocessors)
        # 检查 dataset_id 目录
        if dataset_id:
            encoder_dir = DATA_DIR / 'encoders' / str(dataset_id)
            preprocessor_dir = DATA_DIR / 'preprocessors' / str(dataset_id)
            
            for dir_path in [encoder_dir, preprocessor_dir]:
                if dir_path.exists() and dir_path.is_dir():
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            print(f"已删除空目录: {dir_path}")
                    except OSError as e:
                        print(f"清理空目录失败 {dir_path}: {e}")
        
        # 检查 workflow_id 目录 (新增)
        if workflow_id:
            wf_encoder_dir = DATA_DIR / 'encoders' / str(workflow_id)
            wf_preprocessor_dir = DATA_DIR / 'preprocessors' / str(workflow_id)
            
            for dir_path in [wf_encoder_dir, wf_preprocessor_dir]:
                if dir_path.exists() and dir_path.is_dir():
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            print(f"已删除工作流组件目录: {dir_path}")
                    except OSError as e:
                        print(f"清理工作流目录失败 {dir_path}: {e}")

        # 6. 删除数据库记录
        self.execute_update('DELETE FROM preprocessing_components WHERE model_id = ?', (model_id,))
        self.execute_update('DELETE FROM training_results WHERE model_id = ?', (model_id,))
        self.execute_update('DELETE FROM predictions WHERE model_id = ?', (model_id,))
        
        # 删除模型记录
        rows_affected = self.execute_update('DELETE FROM models WHERE id = ?', (model_id,))

        # 7. 如果需要，删除关联的中间数据集
        if should_delete_dataset:
            try:
                # 此时模型已被删除，可以安全删除数据集
                from backend.services.data_service import DataService
                DataService().delete_dataset(dataset_id)
                print(f"已级联删除中间数据集: {dataset_id}")
            except Exception as e:
                print(f"删除处理后数据集失败: {e}")
                # 尝试直接数据库删除作为后备
                self.delete_dataset(dataset_id)
                
        return rows_affected > 0
    
    # Preprocessing Components相关方法
    def create_preprocessing_component(self, model_id: int, component_type: str,
                                      component_name: str, file_path: str,
                                      applied_columns: List[str], configuration: Dict,
                                      training_statistics: Optional[Dict] = None) -> int:
        """创建预处理组件记录"""
        query = '''
            INSERT INTO preprocessing_components (model_id, component_type, component_name,
                                                 file_path, applied_columns, configuration,
                                                 training_statistics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        return self.execute_insert(query, (
            model_id, component_type, component_name, file_path,
            json.dumps(applied_columns), json.dumps(configuration),
            json.dumps(training_statistics) if training_statistics else None
        ))
    
    def get_preprocessing_components(self, model_id: int) -> List[Dict]:
        """获取模型的所有预处理组件"""
        query = 'SELECT * FROM preprocessing_components WHERE model_id = ?'
        components = self.execute_query(query, (model_id,))
        for comp in components:
            comp['applied_columns'] = json.loads(comp['applied_columns'])
            comp['configuration'] = json.loads(comp['configuration'])
            if comp['training_statistics']:
                comp['training_statistics'] = json.loads(comp['training_statistics'])
        return components
    
    # Training Results相关方法
    def create_training_result(self, model_id: int, computed_results: Dict,
                              visualization_data: Dict, training_log: Optional[str] = None) -> int:
        """创建训练结果记录"""
        query = '''
            INSERT INTO training_results (model_id, computed_results, visualization_data, training_log)
            VALUES (?, ?, ?, ?)
        '''
        return self.execute_insert(query, (
            model_id, json.dumps(computed_results), json.dumps(visualization_data), training_log
        ))
    
    def get_training_result(self, model_id: int) -> Optional[Dict]:
        """获取训练结果"""
        query = 'SELECT * FROM training_results WHERE model_id = ?'
        results = self.execute_query(query, (model_id,))
        if results:
            result = results[0]
            result['computed_results'] = json.loads(result['computed_results'])
            result['visualization_data'] = json.loads(result['visualization_data'])
            return result
        return None
    
    # Predictions相关方法
    def create_prediction(self, model_id: int, input_dataset_id: int, output_file_path: str,
                         workflow_id: Optional[int] = None, used_encoder: bool = False,
                         used_preprocessors: Optional[List[int]] = None,
                         prediction_summary: Optional[Dict] = None) -> int:
        """创建预测记录"""
        query = '''
            INSERT INTO predictions (model_id, workflow_id, input_dataset_id, output_file_path,
                                   used_encoder, used_preprocessors, prediction_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        return self.execute_insert(query, (
            model_id, workflow_id, input_dataset_id, output_file_path, used_encoder,
            json.dumps(used_preprocessors) if used_preprocessors else None,
            json.dumps(prediction_summary) if prediction_summary else None
        ))
    
    def get_prediction(self, prediction_id: int) -> Optional[Dict]:
        """获取预测记录"""
        query = 'SELECT * FROM predictions WHERE id = ?'
        results = self.execute_query(query, (prediction_id,))
        if results:
            pred = results[0]
            if pred['used_preprocessors']:
                pred['used_preprocessors'] = json.loads(pred['used_preprocessors'])
            if pred['prediction_summary']:
                pred['prediction_summary'] = json.loads(pred['prediction_summary'])
            return pred
        return None
    
    def get_model_predictions(self, model_id: int) -> List[Dict]:
        """获取模型的所有预测记录"""
        query = 'SELECT * FROM predictions WHERE model_id = ? ORDER BY created_at DESC'
        predictions = self.execute_query(query, (model_id,))
        for pred in predictions:
            if pred['used_preprocessors']:
                pred['used_preprocessors'] = json.loads(pred['used_preprocessors'])
            if pred['prediction_summary']:
                pred['prediction_summary'] = json.loads(pred['prediction_summary'])
        return predictions
    
    def get_predictions(self, model_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """
        获取预测历史
        
        Args:
            model_id: 模型ID,None表示获取所有
            limit: 返回数量限制
            
        Returns:
            预测历史列表
        """
        if model_id is not None:
            query = 'SELECT * FROM predictions WHERE model_id = ? ORDER BY created_at DESC LIMIT ?'
            predictions = self.execute_query(query, (model_id, limit))
        else:
            query = 'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?'
            predictions = self.execute_query(query, (limit,))
        
        for pred in predictions:
            if pred.get('used_preprocessors'):
                pred['used_preprocessors'] = json.loads(pred['used_preprocessors'])
            if pred.get('prediction_summary'):
                pred['prediction_summary'] = json.loads(pred['prediction_summary'])
        return predictions

    def delete_prediction(self, prediction_id: int) -> bool:
        """删除预测记录"""
        prediction = self.get_prediction(prediction_id)
        if prediction:
            # 删除文件
            if prediction.get('output_file_path') and os.path.exists(prediction['output_file_path']):
                try:
                    os.remove(prediction['output_file_path'])
                except OSError as e:
                    print(f"删除预测结果文件失败: {e}")
            
            # 删除记录
            rows = self.execute_update('DELETE FROM predictions WHERE id = ?', (prediction_id,))
            return rows > 0
        return False
