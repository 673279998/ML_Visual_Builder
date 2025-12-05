"""
模型训练服务
"""
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Tuple, Optional
import shutil
from backend.config import MODEL_DIR, DATA_DIR
from backend.database.db_manager import DatabaseManager
from backend.algorithms.algorithm_factory import AlgorithmFactory
from backend.services.data_service import DataService
from backend.services.preprocessing_service import PreprocessingService
from backend.result_generators import ResultGeneratorFactory
from backend.utils.progress_tracker import progress_tracker
from backend.utils.json_utils import sanitize_for_json

RESULTS_DIR = DATA_DIR / 'training_results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class TrainingService:
    """模型训练服务类"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.data_service = DataService()
        
    def _save_training_results(self, model_id: int, metrics: Dict, visualizations: Dict, 
                             algorithm_name: str, dataset_id: int, feature_importance: Dict = None):
        """保存训练结果到文件"""
        try:
            # 预处理数据，清理NaN和numpy类型
            sanitized_metrics = sanitize_for_json(metrics)
            sanitized_visualizations = sanitize_for_json(visualizations)
            sanitized_feature_importance = sanitize_for_json(feature_importance)
            
            result_data = {
                'model_id': model_id,
                'algorithm': algorithm_name,
                'dataset_id': dataset_id,
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics': sanitized_metrics,
                'feature_importance': sanitized_feature_importance,
                'visualizations': sanitized_visualizations
            }
            
            filepath = RESULTS_DIR / f"results_{model_id}_{algorithm_name}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            # 保存到数据库
            self.db.create_training_result(
                model_id=model_id,
                computed_results=sanitized_metrics,
                visualization_data=sanitized_visualizations,
                training_log=f"Model {algorithm_name} trained on dataset {dataset_id}"
            )
                
            return str(filepath)
        except Exception as e:
            print(f"保存训练结果文件失败: {str(e)}")
            return None
    
    def _collect_dataset_components(self, dataset_id: int) -> List[Dict]:
        """
        回溯数据集的处理历史，收集所有相关的预处理组件
        """
        components = []
        current_id = dataset_id
        
        # 循环回溯直到原始数据集
        while True:
            dataset = self.db.get_dataset(current_id)
            if not dataset or not dataset.get('source_dataset_id'):
                break
                
            # 查询产生此数据集的组件 (from dataset_encoders)
            # dataset_encoders linking source_id -> current_id
            # We query by encoded_dataset_id = current_id
            encoders = self.db.execute_query(
                "SELECT * FROM dataset_encoders WHERE encoded_dataset_id = ?", 
                (current_id,)
            )
            
            # 处理查询到的组件
            for enc in encoders:
                # Parse stored JSONs
                try:
                    summary = json.loads(enc['encoding_summary'])
                    col_map = json.loads(enc['column_mappings'])
                    
                    # Extract type and config
                    c_type = summary.get('type', 'unknown')
                    details = summary.get('details', {})
                    
                    # Construct component dict for preprocessing_components table
                    comp_entry = {
                        'type': c_type,
                        'name': enc['name'],
                        'path': enc['file_path'],
                        'columns': col_map.get('columns', []),
                        'config': details # or some config from details
                    }
                    
                    # 插入到列表开头，因为我们是倒序回溯，但应用顺序应该是正序
                    components.insert(0, comp_entry)
                    
                except Exception as e:
                    print(f"Error parsing component {enc['id']}: {e}")

            current_id = dataset['source_dataset_id']
            
        return components
    
    def _apply_preprocessing_components(self, df: pd.DataFrame, components: List[Dict], 
                                      preprocessor: 'PreprocessingService') -> pd.DataFrame:
        """
        应用预处理组件到数据
        
        Args:
            df: 原始数据
            components: 预处理组件列表
            preprocessor: 预处理服务实例
            
        Returns:
            处理后的数据
        """
        from backend.encoding.label_encoder import LabelEncoder
        from backend.encoding.one_hot_encoder import OneHotEncoder
        from backend.encoding.ordinal_encoder import OrdinalEncoder
        from backend.encoding.target_encoder import TargetEncoder
        from backend.encoding.frequency_encoder import FrequencyEncoder
        from backend.encoding.binary_encoder import BinaryEncoder
        from backend.encoding.hash_encoder import HashEncoder
        import joblib
        
        result_df = df.copy()
        
        # 按类型分组组件，确保正确的执行顺序：编码器 -> 填充器 -> 缩放器
        encoders = [c for c in components if c.get('type') == 'encoder']
        imputers = [c for c in components if c.get('type') == 'imputer']
        scalers = [c for c in components if c.get('type') == 'scaler']
        
        # 1. 应用编码器
        for encoder_comp in encoders:
            try:
                encoder_path = encoder_comp.get('path')
                # 处理相对路径
                if encoder_path and not os.path.isabs(encoder_path):
                    from backend.config import BASE_DIR
                    encoder_path = os.path.join(BASE_DIR, encoder_path)
                
                if not encoder_path or not os.path.exists(encoder_path):
                    print(f"警告: 编码器路径不存在: {encoder_path}")
                    continue
                    
                encoder = joblib.load(encoder_path)
                columns = encoder_comp.get('columns', [])
                config = encoder_comp.get('config', {})
                method = config.get('method', 'unknown')
                
                print(f"应用编码器: {encoder_comp.get('name')}, 方法: {method}, 列: {columns}")
                
                for col in columns:
                    if col in result_df.columns:
                        try:
                            if method == 'onehot':
                                # OneHot编码会产生新列
                                transformed = encoder.transform(result_df[[col]])
                                # 删除原列
                                result_df = result_df.drop(columns=[col])
                                # 添加新列
                                result_df = pd.concat([result_df, transformed], axis=1)
                                print(f"  - 列 {col}: OneHot编码完成，生成 {transformed.shape[1]} 个新列")
                            else:
                                # 其他编码替换原列
                                result_df[col] = encoder.transform(result_df[[col]]).iloc[:, 0]
                                print(f"  - 列 {col}: {method}编码完成")
                        except Exception as col_error:
                            print(f"  - 列 {col} 编码失败: {col_error}")
                    else:
                        print(f"  - 列 {col} 不在数据集中，跳过")
            except Exception as e:
                print(f"警告: 应用编码器 {encoder_comp.get('name')} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. 应用填充器
        for imputer_comp in imputers:
            try:
                imputer_path = imputer_comp.get('path')
                # 处理相对路径
                if imputer_path and not os.path.isabs(imputer_path):
                    from backend.config import BASE_DIR
                    imputer_path = os.path.join(BASE_DIR, imputer_path)
                
                if not imputer_path or not os.path.exists(imputer_path):
                    print(f"警告: 填充器路径不存在: {imputer_path}")
                    continue
                    
                imputer = joblib.load(imputer_path)
                columns = imputer_comp.get('columns', [])
                
                print(f"应用填充器: {imputer_comp.get('name')}, 列: {columns}")
                
                # 只处理数值列
                numeric_cols = [col for col in columns if col in result_df.columns and 
                               pd.api.types.is_numeric_dtype(result_df[col])]
                
                if numeric_cols:
                    result_df[numeric_cols] = imputer.transform(result_df[numeric_cols])
                    print(f"  - 填充 {len(numeric_cols)} 个数值列")
                else:
                    print(f"  - 没有找到可填充的数值列")
            except Exception as e:
                print(f"警告: 应用填充器 {imputer_comp.get('name')} 失败: {e}")
        
        # 3. 应用缩放器
        for scaler_comp in scalers:
            try:
                scaler_path = scaler_comp.get('path')
                # 处理相对路径
                if scaler_path and not os.path.isabs(scaler_path):
                    from backend.config import BASE_DIR
                    scaler_path = os.path.join(BASE_DIR, scaler_path)
                
                if not scaler_path or not os.path.exists(scaler_path):
                    print(f"警告: 缩放器路径不存在: {scaler_path}")
                    continue
                    
                scaler = joblib.load(scaler_path)
                columns = scaler_comp.get('columns', [])
                
                print(f"应用缩放器: {scaler_comp.get('name')}, 列: {columns}")
                
                # 只处理数值列
                numeric_cols = [col for col in columns if col in result_df.columns and 
                               pd.api.types.is_numeric_dtype(result_df[col])]
                
                if numeric_cols:
                    result_df[numeric_cols] = scaler.transform(result_df[numeric_cols])
                    print(f"  - 缩放 {len(numeric_cols)} 个数值列")
                else:
                    print(f"  - 没有找到可缩放的数值列")
            except Exception as e:
                print(f"警告: 应用缩放器 {scaler_comp.get('name')} 失败: {e}")
        
        return result_df

    def train_model(self, dataset_id: int, algorithm_name: str, 
                   target_columns: List[str], feature_columns: List[str] = None,
                   test_size: float = 0.2, hyperparameters: Dict = None,
                   preprocessing_config: Dict = None,
                   preprocessing_components: List[Dict] = None,
                   task_id: Optional[str] = None,
                   original_dataset_id: Optional[int] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            dataset_id: 数据集ID
            algorithm_name: 算法名称
            target_columns: 目标变量列名列表
            feature_columns: 特征列名列表(None表示使用所有非目标列)
            test_size: 测试集比例
            hyperparameters: 超参数字典
            preprocessing_config: 预处理配置
            preprocessing_components: 外部传入的预处理组件列表
            task_id: 任务ID（用于进度追踪）
            original_dataset_id: 原始数据集ID (用于生成正确的输入需求)
            
        Returns:
            训练结果
        """
        # 加载数据
        if task_id:
            progress_tracker.update_progress(task_id, current_step=1, message='准备数据...')
        
        # 获取数据集信息
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")

        # 创建算法实例 (提前创建以获取算法类型)
        algorithm = AlgorithmFactory.create_algorithm(algorithm_name)

        # 确定目标列 (提前确定以指导预处理)
        if not target_columns:
            columns_config = self.db.get_columns(dataset_id)
            target_columns = [col['column_name'] for col in columns_config if col['is_target']]
            
            if not target_columns:
                raise ValueError("未指定目标列，且数据集未配置默认目标列")
            
        # 如果未指定 original_dataset_id，尝试追溯
        if original_dataset_id is None and dataset.get('source_dataset_id'):
            current_ds = dataset
            while current_ds.get('source_dataset_id'):
                source_id = current_ds['source_dataset_id']
                source_ds = self.db.get_dataset(source_id)
                if source_ds:
                    current_ds = source_ds
                else:
                    break
            if current_ds['id'] != dataset_id:
                original_dataset_id = current_ds['id']

        # 读取数据
        df = self.data_service._read_file(dataset['file_path'], dataset['file_format'])
        
        # 预处理
        preprocessor = PreprocessingService()
        
        # 自动处理日期列 (防止直接训练时因日期字符串报错)
        df = preprocessor._handle_datetime_columns(df)
        
        components_to_save = []
        
        # 1. 收集数据集历史中的预处理组件 (新增逻辑)
        history_components = self._collect_dataset_components(dataset_id)
        if history_components:
            components_to_save.extend(history_components)
        
        if preprocessing_components:
            components_to_save.extend(preprocessing_components)
        
        # 应用预处理组件
        print(f"训练服务: 准备应用预处理组件，组件数量: {len(components_to_save)}")
        if components_to_save:
            print(f"训练服务: 组件详情: {[c.get('type') + '_' + c.get('name', 'unknown') for c in components_to_save]}")
            df = self._apply_preprocessing_components(df, components_to_save, preprocessor)
            print(f"训练服务: 应用预处理组件后，数据列: {list(df.columns)}")
            print(f"训练服务: 数据形状: {df.shape}")
        else:
            print(f"训练服务: 没有预处理组件需要应用")
        
        if preprocessing_config:
            # 1. 缺失值处理
            if 'imputation' in preprocessing_config:
                impute_config = preprocessing_config['imputation']
                df, info = preprocessor.handle_missing_values(
                    df, 
                    strategy=impute_config.get('strategy', 'mean'),
                    columns=impute_config.get('columns'),
                    dataset_id=dataset_id
                )
                if info.get('saved_imputers'):
                    for imputer_info in info['saved_imputers']:
                        components_to_save.append({
                            'type': 'imputer',
                            'name': f"imputer_{imputer_info['strategy']}",
                            'path': imputer_info['path'],
                            'columns': imputer_info['columns'],
                            'config': impute_config
                        })
            
            # 2. 自动编码
            if 'encoding' in preprocessing_config:
                encode_configs = preprocessing_config['encoding']
                # 格式: [{'column_name': 'col1', 'encoding_method': 'label'}]
                df, info = preprocessor.auto_encode_features(
                    df,
                    column_configs=encode_configs,
                    dataset_id=dataset_id
                )
                
                # 检查是否有编码错误
                if info.get('errors'):
                    error_msg = "; ".join(info['errors'])
                    raise ValueError(f"特征编码失败: {error_msg}")
                    
                if info.get('saved_encoders'):
                    for encoder_info in info['saved_encoders']:
                        components_to_save.append({
                            'type': 'encoder',
                            'name': f"encoder_{encoder_info['column']}_{encoder_info['method']}",
                            'path': encoder_info['path'],
                            'columns': [encoder_info['column']],
                            'config': {'method': encoder_info['method']}
                        })

            # 3. 特征缩放
            if 'scaling' in preprocessing_config:
                scale_config = preprocessing_config['scaling']
                scale_method = scale_config.get('method', 'standard')
                specified_cols = scale_config.get('columns')
                
                # 确定要缩放的列
                if specified_cols:
                    cols_to_scale = specified_cols
                else:
                    cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
                
                y_scaler = None
                
                if algorithm.algorithm_type == 'classification':
                    # 分类任务：排除目标列，防止目标被缩放成连续值
                    cols_to_scale = [c for c in cols_to_scale if c not in target_columns]
                    if cols_to_scale:
                        df, info = preprocessor.scale_features(
                            df, method=scale_method, columns=cols_to_scale, dataset_id=dataset_id
                        )
                        if info.get('saved_scaler'):
                            scaler_info = info['saved_scaler']
                            components_to_save.append({
                                'type': 'scaler',
                                'name': f"scaler_{scaler_info['method']}",
                                'path': scaler_info['path'],
                                'columns': scaler_info['columns'],
                                'config': scale_config
                            })

                elif algorithm.algorithm_type == 'regression':
                    # 回归任务：分离特征和目标，分别缩放
                    cols_to_scale_X = [c for c in cols_to_scale if c not in target_columns]
                    cols_to_scale_y = [c for c in cols_to_scale if c in target_columns]
                    
                    # 缩放特征 (X)
                    if cols_to_scale_X:
                        df, info_X = preprocessor.scale_features(
                            df, method=scale_method, columns=cols_to_scale_X, dataset_id=dataset_id, suffix="X"
                        )
                        if info_X.get('saved_scaler'):
                            scaler_info = info_X['saved_scaler']
                            components_to_save.append({
                                'type': 'scaler',
                                'name': f"scaler_X_{scaler_info['method']}",
                                'path': scaler_info['path'],
                                'columns': scaler_info['columns'],
                                'config': scale_config
                            })
                    
                    # 缩放目标 (y)
                    if cols_to_scale_y:
                        df, info_y = preprocessor.scale_features(
                            df, method=scale_method, columns=cols_to_scale_y, dataset_id=dataset_id, suffix="y"
                        )
                        if info_y.get('saved_scaler'):
                            scaler_info = info_y['saved_scaler']
                            components_to_save.append({
                                'type': 'scaler',
                                'name': f"scaler_y_{scaler_info['method']}",
                                'path': scaler_info['path'],
                                'columns': scaler_info['columns'],
                                'config': scale_config
                            })
                            # 保存y_scaler用于后续反缩放
                            if 'scaler_key' in info_y:
                                y_scaler = preprocessor.scalers.get(info_y['scaler_key'])

        # 准备X和y (使用处理后的df)
        # 目标列已在上方确定
        
        # 分离特征和目标
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in target_columns]

        # [Safety Check] Ensure all feature columns are numeric
        print(f"训练服务: 验证特征列，特征列: {feature_columns}")
        print(f"训练服务: 数据列: {list(df.columns)}")
        print(f"训练服务: 数据类型: {[str(df[col].dtype) for col in df.columns]}")
        
        invalid_feature_columns = []
        for col in feature_columns:
            if col not in df.columns:
                print(f"训练服务: 警告: 特征列 {col} 不在数据集中")
                continue
                
            dtype_str = str(df[col].dtype)
            if df[col].dtype == 'object' or dtype_str == 'category':
                print(f"训练服务: 特征列 {col} 包含非数值数据，类型: {dtype_str}")
                invalid_feature_columns.append(col)
        
        if invalid_feature_columns:
            error_details = f"Invalid columns: {invalid_feature_columns}. Available columns in dataset: {list(df.columns)}. "
            if preprocessing_config and 'encoding' in preprocessing_config:
                error_details += f"Encoding config provided for: {[c['column_name'] for c in preprocessing_config['encoding']]}."
            else:
                error_details += "No encoding config provided."
                
            raise ValueError(f"特征列 {', '.join(invalid_feature_columns)} 包含非数值数据（如字符串）。请检查数据预处理配置，确保对所有分类变量进行了编码（如 Label, One-Hot）或将其排除。\n调试信息: {error_details}")
        
        X = df[feature_columns].values
        y = df[target_columns].values
        
        # 如果只有一个目标列,展平
        if y.shape[1] == 1:
            y = y.ravel()
            
        feature_names = feature_columns
        
        # 创建算法实例
        # algorithm = AlgorithmFactory.create_algorithm(algorithm_name) # Already created above

        # 针对分类任务的特殊处理：使用LabelEncoder确保目标变量为离散整数
        # 解决 "Unknown label type: continuous" 错误 (当目标变量为float类型 0.0, 1.0 时)
        if algorithm.algorithm_type == 'classification':
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # 确保y是1D
                if len(y.shape) > 1:
                    y = y.ravel()
                y = le.fit_transform(y)
                print(f"DEBUG: 已对分类目标变量进行LabelEncoding转换, Classes: {le.classes_}")
                
                # 保存LabelEncoder以便预测时还原
                encoder_dir = DATA_DIR / 'encoders' / str(dataset_id)
                if not encoder_dir.exists():
                    encoder_dir.mkdir(parents=True, exist_ok=True)
                    
                le_filename = f"target_label_encoder_{algorithm_name}_{dataset_id}.pkl"
                le_path = encoder_dir / le_filename
                joblib.dump(le, le_path)
                
                # 添加到组件列表以便保存到数据库
                # 注意：这里我们使用 target_columns 作为 applied_columns
                components_to_save.append({
                    'type': 'encoder', # 使用 'encoder' 类型，PredictionService 会识别它
                    'name': f"target_encoder_{algorithm_name}",
                    'path': str(le_path), # 保存绝对路径或相对路径
                    'columns': target_columns,
                    'config': {'method': 'label_encoder', 'is_target': True}
                })
                
            except Exception as e:
                print(f"WARNING: LabelEncoding转换或保存失败: {e}")
                import traceback
                traceback.print_exc()

        # 划分训练集和测试集
        if task_id:
            progress_tracker.update_progress(task_id, current_step=2, message='划分训练集和测试集...')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 训练模型
        if task_id:
            progress_tracker.update_progress(task_id, current_step=3, message=f'训练{algorithm.algorithm_name}模型...')
        
        if hyperparameters is None:
            hyperparameters = {}
        algorithm.train(X_train, y_train, **hyperparameters)
        
        # 预测
        if task_id:
            progress_tracker.update_progress(task_id, current_step=4, message='评估模型性能...')
        
        y_pred = algorithm.predict(X_test)
        
        # 反缩放 (如果是回归任务且使用了目标变量缩放)
        if algorithm.algorithm_type == 'regression' and 'y_scaler' in locals() and y_scaler:
            try:
                # y_test and y_pred might be 1D arrays, scaler expects 2D
                y_test_reshaped = y_test.reshape(-1, 1)
                y_pred_reshaped = y_pred.reshape(-1, 1)
                
                y_test = y_scaler.inverse_transform(y_test_reshaped).ravel()
                y_pred = y_scaler.inverse_transform(y_pred_reshaped).ravel()
                
                print("DEBUG: 已对回归任务的预测结果进行反缩放")
            except Exception as e:
                print(f"WARNING: 反缩放失败: {e}")

        # 计算性能指标
        metrics = self._calculate_metrics(algorithm.algorithm_type, y_test, y_pred, algorithm, X_test)
        
        # 生成完整结果(指标+可视化)
        complete_results = self._generate_complete_results(
            algorithm.algorithm_type, y_test, y_pred, algorithm.model, X_test, feature_names
        )
        
        # 保存模型
        if task_id:
            progress_tracker.update_progress(task_id, current_step=5, message='保存模型...')
        
        model_path = self._save_model(algorithm, dataset_id)
        
        # 生成数据集Schema (仅包含特征列)
        dataset_schema = {col: str(df[col].dtype) for col in feature_columns}
        
        # 构建输入需求
        # 如果提供了original_dataset_id，则使用原始数据集的列作为输入需求
        if original_dataset_id:
            original_columns_config = self.db.get_columns(original_dataset_id)
            # 筛选出非目标且非排除的列，同时排除传入的 target_columns
            original_features = [
                col['column_name'] for col in original_columns_config 
                if not col['is_target'] 
                and not col.get('is_excluded', False)
                and col['column_name'] not in target_columns
            ]
            input_requirements = {
                'features': original_features,
                'target': target_columns,
                'required_columns': original_features,
                'original_dataset_id': original_dataset_id
            }
        else:
            input_requirements = {
                'features': feature_columns,
                'target': target_columns,
                'required_columns': feature_columns  # 必须包含的列
            }
        
        # 获取特征重要性
        feature_importance = algorithm.get_feature_importance()
        if feature_importance and feature_names:
            # 替换特征名称
            feature_importance = {feature_names[int(k.split('_')[1])]: v 
                                for k, v in feature_importance.items()}
        
        # 保存到数据库
        model_id = self.db.create_model(
            name=f"{algorithm_name}_{dataset_id}",
            algorithm_type=algorithm.algorithm_type,
            algorithm_name=algorithm_name,
            dataset_id=dataset_id,
            model_file_path=model_path,
            hyperparameters=sanitize_for_json(hyperparameters),
            performance_metrics=sanitize_for_json(metrics),
            dataset_schema=sanitize_for_json(dataset_schema),
            input_requirements=sanitize_for_json(input_requirements),
            feature_importance=sanitize_for_json(feature_importance)
        )
        
        # 保存预处理组件到数据库
        # 直接引用原始组件路径，不再复制到 model_components 目录
        for comp in components_to_save:
            self.db.create_preprocessing_component(
                model_id=model_id,
                component_type=comp['type'],
                component_name=comp['name'],
                file_path=comp['path'],
                applied_columns=comp['columns'],
                configuration=comp['config']
            )
        
        # 保存结果到文件
        self._save_training_results(
            model_id=model_id,
            metrics=metrics,
            visualizations=complete_results,
            algorithm_name=algorithm_name,
            dataset_id=dataset_id,
            feature_importance=feature_importance
        )
        
        return {
            'model_id': model_id,
            'algorithm_name': algorithm_name,
            'algorithm_type': algorithm.algorithm_type,
            'performance_metrics': sanitize_for_json(metrics),
            'feature_importance': sanitize_for_json(feature_importance),
            'dataset_schema': sanitize_for_json(dataset_schema),
            'complete_results': sanitize_for_json(complete_results)  # 完整结果(包含可视化数据)
        }
    
    def _prepare_data(self, dataset_id: int, target_columns: List[str], 
                     feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练数据"""
        # 获取数据集
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        
        # 读取数据
        df = self.data_service._read_file(dataset['file_path'], dataset['file_format'])
        
        # 如果未指定目标列，尝试从数据集配置中获取
        if not target_columns:
            columns_config = self.db.get_columns(dataset_id)
            target_columns = [col['column_name'] for col in columns_config if col['is_target']]
            
            if not target_columns:
                raise ValueError("未指定目标列，且数据集未配置默认目标列")
        
        # 分离特征和目标
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in target_columns]
        
        X = df[feature_columns].values
        y = df[target_columns].values
        
        # 如果只有一个目标列,展平
        if y.shape[1] == 1:
            y = y.ravel()
        
        return X, y, feature_columns
    
    def _calculate_metrics(self, algorithm_type: str, y_true: np.ndarray, 
                          y_pred: np.ndarray, algorithm, X_test: np.ndarray) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {}
        
        if algorithm_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            # 尝试计算ROC-AUC
            try:
                if hasattr(algorithm, 'predict_proba'):
                    y_prob = algorithm.predict_proba(X_test)
                    if len(np.unique(y_true)) == 2:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
                    else:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
            except:
                pass
                
        elif algorithm_type == 'regression':
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    def _save_model(self, algorithm, dataset_id: int) -> str:
        """保存模型"""
        filename = f"model_{dataset_id}_{algorithm.algorithm_name}.pkl"
        file_path = os.path.join(MODEL_DIR, filename)
        joblib.dump(algorithm.model, file_path)
        return file_path
    
    def _generate_dataset_schema(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], target_names: List[str]) -> Dict[str, Any]:
        """生成数据集格式Schema"""
        schema = {
            'feature_count': X.shape[1],
            'target_count': 1 if len(y.shape) == 1 else y.shape[1],
            'features': [],
            'targets': [],
            'row_count_range': [1, 100000]
        }
        
        # 特征信息
        for i, name in enumerate(feature_names):
            feature_col = X[:, i]
            schema['features'].append({
                'name': name,
                'type': 'numeric',
                'range': [float(np.min(feature_col)), float(np.max(feature_col))],
                'mean': float(np.mean(feature_col)),
                'std': float(np.std(feature_col)),
                'sample_values': [float(v) for v in feature_col[:3]]
            })
        
        # 目标变量信息
        for i, name in enumerate(target_names):
            if len(y.shape) == 1:
                target_col = y
            else:
                target_col = y[:, i]
            
            # 检查是否为数值类型
            try:
                # 尝试转换为浮点数来判断是否为数值型
                target_col_float = target_col.astype(float)
                target_info = {
                    'name': name,
                    'type': 'numeric',
                    'range': [float(np.min(target_col_float)), float(np.max(target_col_float))]
                }
            except (ValueError, TypeError):
                # 非数值类型，作为分类型处理
                unique_values = np.unique(target_col)
                target_info = {
                    'name': name,
                    'type': 'categorical',
                    'unique_values': [str(v) for v in unique_values[:50]]  # 限制数量
                }
            
            schema['targets'].append(target_info)
        
        return schema
    
    def _generate_input_requirements(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """生成输入数据要求"""
        requirements = {
            'required_columns': [f['name'] for f in schema['features']],
            'column_requirements': {},
            'row_constraints': {
                'min_rows': schema['row_count_range'][0],
                'max_rows': schema['row_count_range'][1]
            },
            'file_format': 'CSV或Excel,需包含列标题'
        }
        
        for feature in schema['features']:
            requirements['column_requirements'][feature['name']] = {
                'type': feature['type'],
                'constraints': {
                    'min': feature['range'][0],
                    'max': feature['range'][1]
                },
                'description': f"{feature['name']},范围{feature['range'][0]}-{feature['range'][1]}"
            }
        
        return requirements
    
    def _generate_complete_results(self, algorithm_type: str, y_true: np.ndarray, 
                                   y_pred: np.ndarray, model: Any, 
                                   X_test: np.ndarray = None,
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        生成完整结果(指标+可视化)
        
        Args:
            algorithm_type: 算法类型
            y_true: 真实标签
            y_pred: 预测标签
            model: 训练好的模型
            X_test: 测试特征数据
            feature_names: 特征名称列表
            
        Returns:
            完整结果字典
        """
        try:
            # 创建结果生成器
            generator = ResultGeneratorFactory.create_generator(algorithm_type)
            
            # 生成完整结果
            complete_results = generator.generate_complete_results(
                y_true=y_true,
                y_pred=y_pred,
                model=model,
                X_test=X_test,
                feature_names=feature_names
            )
            
            return complete_results
            
        except Exception as e:
            return {
                'error': f'生成结果失败: {str(e)}',
                'result_type': algorithm_type,
                'metrics': {},
                'visualizations': {}
            }
