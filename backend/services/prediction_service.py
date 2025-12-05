"""
预测服务
Prediction Service
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import joblib
from datetime import datetime
import logging
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder

from backend.config import MODELS_DIR, PREDICTIONS_DIR
from backend.database.db_manager import DatabaseManager
from backend.algorithms.algorithm_factory import AlgorithmFactory
from backend.services.preprocessing_service import PreprocessingService

logger = logging.getLogger(__name__)


class PredictionService:
    """预测服务"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def predict(
        self,
        model_id: int,
        input_data: pd.DataFrame,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        使用模型进行预测
        
        Args:
            model_id: 模型ID
            input_data: 输入数据
            return_probabilities: 是否返回概率(仅分类模型)
            
        Returns:
            预测结果
        """
        # 获取模型信息
        model_info = self.db_manager.get_model(model_id)
        if not model_info:
            raise ValueError(f"模型不存在: {model_id}")
        
        # 加载模型
        model_path = Path(model_info['model_file_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        algorithm = joblib.load(model_path)
        
        # 智能列名映射 (处理空格/下划线不一致的情况)
        input_requirements = model_info.get('input_requirements', {})
        required_features = input_requirements.get('required_columns', input_requirements.get('features', []))
        
        if required_features:
            # 创建归一化映射: normalized_name -> original_name
            norm_input_map = {str(c).strip().replace(' ', '_'): c for c in input_data.columns}
            
            rename_map = {}
            for req in required_features:
                if req not in input_data.columns:
                    # 尝试归一化匹配
                    req_norm = str(req).strip().replace(' ', '_')
                    if req_norm in norm_input_map:
                        current_col = norm_input_map[req_norm]
                        rename_map[current_col] = req
            
            if rename_map:
                input_data.rename(columns=rename_map, inplace=True)
        
        # 验证输入数据
        validation_result = self._validate_input_data(input_data, model_info)
        if not validation_result['valid']:
            # 如果验证失败，但有预处理组件，尝试先预处理再验证（或者在准备数据时处理）
            # 但这里我们先获取预处理组件看看
            components = self.db_manager.get_preprocessing_components(model_id)
            if components:
                # 如果有预处理组件，说明输入数据可能是原始数据，而model_info中的requirements是处理后的
                # 我们应该允许 validation 失败，在 _prepare_input_data 中处理
                pass
            else:
                raise ValueError(f"输入数据验证失败: {validation_result['errors']}")
        
        # 准备数据
        X = self._prepare_input_data(input_data, model_info)
        
        # 进行预测
        predictions = algorithm.predict(X)
        
        # 尝试反转变换（还原为原始数据）
        original_predictions = self._inverse_transform_predictions(predictions, model_id)
        
        result = {
            'model_id': model_id,
            'model_name': model_info['name'],
            'algorithm_type': model_info['algorithm_type'],
            'n_samples': len(predictions),
            'predictions': original_predictions.tolist(),
            'raw_predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 如果是分类模型且需要概率
        if return_probabilities and model_info['algorithm_type'] == 'classification':
            if hasattr(algorithm, 'predict_proba'):
                probabilities = algorithm.predict_proba(X)
                result['probabilities'] = probabilities.tolist()
                
                # 获取类别标签
                if hasattr(algorithm, 'classes_'):
                    result['class_labels'] = algorithm.classes_.tolist()
        
        # 保存预测记录
        prediction_id = self._save_prediction_record(
            model_id=model_id,
            input_data=input_data,
            predictions=predictions,
            probabilities=result.get('probabilities')
        )
        
        result['prediction_id'] = prediction_id
        result['message'] = "预测结果文件已保存"
        
        return result
    
    def batch_predict(
        self,
        model_id: int,
        data_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        批量预测
        
        Args:
            model_id: 模型ID
            data_file: 数据文件路径
            output_file: 输出文件路径
            
        Returns:
            批量预测结果
        """
        # 读取数据
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        elif data_file.endswith('.xlsx'):
            data = pd.read_excel(data_file)
        else:
            raise ValueError(f"不支持的文件格式: {data_file}")
        
        # 进行预测
        result = self.predict(model_id, data, return_probabilities=True)
        
        # 将预测结果添加到数据中
        data['prediction'] = result['predictions']
        if 'probabilities' in result:
            for i, label in enumerate(result.get('class_labels', [])):
                data[f'probability_{label}'] = [p[i] for p in result['probabilities']]
        
        # 保存输出文件
        if output_file:
            output_path = PREDICTIONS_DIR / output_file
            if output_file.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif output_file.endswith('.xlsx'):
                data.to_excel(output_path, index=False)
            
            result['output_file'] = str(output_path)
        
        result['output_data'] = data.to_dict('records')
        result['message'] = "预测结果文件已保存"
        
        return result
    
    def _validate_input_data(
        self,
        data: pd.DataFrame,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            model_info: 模型信息
            
        Returns:
            验证结果
        """
        errors = []
        warnings = []
        
        # 获取模型的输入要求
        input_requirements = model_info.get('input_requirements', {})
        required_features = input_requirements.get('required_columns', input_requirements.get('features', []))
        
        # 检查必需特征
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            errors.append(f"缺少必需特征: {missing_features}")
        
        # 检查额外特征
        extra_features = set(data.columns) - set(required_features)
        if extra_features:
            warnings.append(f"存在额外特征(将被忽略): {extra_features}")
        
        # 检查数据类型
        feature_types = input_requirements.get('feature_types', {})
        for feature, expected_type in feature_types.items():
            if feature in data.columns:
                actual_type = str(data[feature].dtype)
                if expected_type == 'numeric' and actual_type not in ['int64', 'float64']:
                    errors.append(f"特征 '{feature}' 应为数值类型,实际为 {actual_type}")
                elif expected_type == 'categorical' and actual_type not in ['object', 'category']:
                    warnings.append(f"特征 '{feature}' 应为类别类型,实际为 {actual_type}")
        
        # 检查缺失值 - 只检查实际存在的特征
        existing_features = [f for f in required_features if f in data.columns]
        if existing_features:
            missing_counts = data[existing_features].isna().sum()
            if missing_counts.sum() > 0:
                warnings.append(f"存在缺失值: {missing_counts[missing_counts > 0].to_dict()}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _apply_preprocessing(self, data: pd.DataFrame, components: List[Dict]) -> pd.DataFrame:
        df = data.copy()
        preprocessor = PreprocessingService()
        df = preprocessor._handle_datetime_columns(df)

        order_map = {'imputer': 0, 'encoder': 1, 'scaler': 2}
        components_sorted = sorted(components, key=lambda c: order_map.get(c.get('component_type'), 99))

        encoders = [c for c in components_sorted if c.get('component_type') == 'encoder']
        imputers = [c for c in components_sorted if c.get('component_type') == 'imputer']
        scalers = [c for c in components_sorted if c.get('component_type') == 'scaler']

        for comp in imputers:
            c_path = comp['file_path']
            c_cols = comp['applied_columns']
            if c_path and not os.path.isabs(c_path):
                from backend.config import BASE_DIR
                c_path = os.path.join(BASE_DIR, c_path)
            if not c_path or not os.path.exists(c_path):
                logger.error(f"预处理组件文件不存在: {c_path} (组件: {comp.get('component_name')})")
                raise FileNotFoundError(f"预处理组件文件不存在: {c_path}")
            try:
                transformer = joblib.load(c_path)
                missing_cols = [c for c in c_cols if c not in df.columns]
                if missing_cols:
                    logger.warning(f"组件 {comp['component_name']} 缺少输入列: {missing_cols}")
                    continue
                data_part = df[c_cols]
                transformed = transformer.transform(data_part)
                df[c_cols] = transformed
            except Exception as e:
                logger.error(f"应用预处理组件 {comp['component_name']} 失败: {str(e)}")

        # 先应用编码器，因为编码器可能会创建新列
        for comp in encoders:
            c_path = comp['file_path']
            c_cols = comp['applied_columns']
            if c_path and not os.path.isabs(c_path):
                from backend.config import BASE_DIR
                c_path = os.path.join(BASE_DIR, c_path)
            if not c_path or not os.path.exists(c_path):
                logger.error(f"预处理组件文件不存在: {c_path} (组件: {comp.get('component_name')})")
                raise FileNotFoundError(f"预处理组件文件不存在: {c_path}")
            try:
                transformer = joblib.load(c_path)
                for col in c_cols:
                    if col not in df.columns:
                        continue
                    series = df[col]
                    transformed = transformer.transform(series)
                    if hasattr(transformed, 'toarray'):
                        transformed = transformed.toarray()
                    if isinstance(transformed, pd.DataFrame):
                        df = df.drop(columns=[col])
                        df = pd.concat([df, transformed], axis=1)
                    elif isinstance(transformed, np.ndarray):
                        if transformed.ndim == 1:
                            df[col] = transformed
                        else:
                            try:
                                if hasattr(transformer, 'get_feature_names_out'):
                                    new_cols = transformer.get_feature_names_out([col])
                                else:
                                    new_cols = [f"{col}_{i}" for i in range(transformed.shape[1])]
                            except:
                                new_cols = [f"{col}_{i}" for i in range(transformed.shape[1])]
                            encoded_df = pd.DataFrame(transformed, columns=new_cols, index=df.index)
                            df = df.drop(columns=[col])
                            df = pd.concat([df, encoded_df], axis=1)
                    else:
                        df[col] = transformed
            except Exception as e:
                logger.error(f"应用预处理组件 {comp['component_name']} 失败: {str(e)}")

        # 现在应用缩放器，缩放器应该作用于编码后的列
        for comp in scalers:
            c_path = comp['file_path']
            c_cols = comp['applied_columns']
            if c_path and not os.path.isabs(c_path):
                from backend.config import BASE_DIR
                c_path = os.path.join(BASE_DIR, c_path)
            if not c_path or not os.path.exists(c_path):
                logger.error(f"预处理组件文件不存在: {c_path} (组件: {comp.get('component_name')})")
                raise FileNotFoundError(f"预处理组件文件不存在: {c_path}")
            try:
                transformer = joblib.load(c_path)
                
                # 增强的列名匹配逻辑
                # 目的：解决输入数据列名格式（如空格、下划线、大小写）与组件期望不一致的问题
                # 避免因匹配失败导致错误的"补0"操作
                
                actual_cols_in_df = []
                # 创建归一化映射: normalized_name -> actual_name
                # 归一化规则: 转小写，去除空格、下划线、括号
                def normalize_col(name):
                    return str(name).strip().lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', '')
                
                df_col_norm = {normalize_col(c): c for c in df.columns}
                
                missing_cols_log = []
                
                for req_col in c_cols:
                    if req_col in df.columns:
                        actual_cols_in_df.append(req_col)
                    else:
                        # 尝试模糊匹配
                        req_norm = normalize_col(req_col)
                        if req_norm in df_col_norm:
                            found_col = df_col_norm[req_norm]
                            actual_cols_in_df.append(found_col)
                            logger.info(f"缩放器列名模糊匹配: 期望 '{req_col}' -> 实际 '{found_col}'")
                        else:
                            # 确实缺失
                            missing_cols_log.append(req_col)
                            # 仅作为最后的手段填充0 (兼容OneHot稀疏情况)
                            # 但对于数值特征，这通常意味着错误
                            logger.warning(f"缩放器期望列 '{req_col}' 缺失且无法匹配，填充0。这可能导致预测结果严重偏差！")
                            df[req_col] = 0
                            actual_cols_in_df.append(req_col)
                
                if missing_cols_log:
                    logger.warning(f"组件 {comp['component_name']} 存在缺失列: {missing_cols_log}")
                
                # 按照组件期望的顺序提取数据（使用匹配到的实际列名）
                data_part = df[actual_cols_in_df]
                
                # 调试：记录数据形状
                logger.debug(f"缩放器应用: 组件={comp['component_name']}, 数据形状={data_part.shape}")
                
                # 使用 .values 避免 sklearn 检查列名 (只要顺序一致即可)
                # data_part 的列顺序已经通过 actual_cols_in_df 保证与组件期望一致
                transformed = transformer.transform(data_part.values)
                
                # 将转换后的数据写回 DataFrame (更新对应的实际列)
                # 注意：这里会直接修改实际存在的列
                # 如果是新创建的补0列，也会被更新
                
                # 为了避免SettingWithCopyWarning或潜在的赋值问题，我们直接赋值
                # Pandas 允许 df[list_of_cols] = matrix
                df[actual_cols_in_df] = transformed
                
                logger.debug(f"缩放器应用成功: {comp['component_name']}")
                
            except Exception as e:
                logger.error(f"应用预处理组件 {comp['component_name']} 失败: {str(e)}")

        return df

    def _prepare_input_data(
        self,
        data: pd.DataFrame,
        model_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        准备输入数据
        
        Args:
            data: 原始数据
            model_info: 模型信息
            
        Returns:
            准备好的数据数组
        """
        logger.info(f"Preparing input data. Shape: {data.shape}")
        logger.info(f"Input columns: {data.columns.tolist()}")
        logger.info(f"Input data head (raw): {data.iloc[0].to_dict() if not data.empty else 'Empty'}")

        # 1. 获取预处理组件
        model_id = model_info['id']
        components = self.db_manager.get_preprocessing_components(model_id)
        logger.info(f"Found {len(components)} preprocessing components for model {model_id}")
        
        # 2. 应用预处理
        if components:
            data = self._apply_preprocessing(data, components)
            logger.info(f"Data after preprocessing: {data.iloc[0].to_dict() if not data.empty else 'Empty'}")
        
        # 3. 选择模型需要的特征
        # 注意：这里的 required_features 应该是经过预处理后的特征列表
        # 如果 input_requirements 存储的是原始特征，我们需要另一种方式知道"最终特征"
        # 通常算法模型(sklearn)有 feature_names_in_ 属性，或者我们可以从 dataset_schema 中推断
        # 这里的 input_requirements 可能是我们在 training_service 中保存的
        
        # 如果我们在 training_service 中把 input_requirements 改成了原始特征
        # 那么我们需要从哪里获取 经过处理后的特征名？
        # model_info['dataset_schema'] 通常存储的是训练数据的schema (即处理后的)
        
        dataset_schema = model_info.get('dataset_schema', {})
        # dataset_schema 格式: {'col1': 'float', ...}
        # 或者是 list of dict
        
        if isinstance(dataset_schema, list):
             model_features = [col['name'] for col in dataset_schema]
        elif isinstance(dataset_schema, dict):
             model_features = list(dataset_schema.keys())
        else:
             model_features = []

        # 如果没有 dataset_schema，尝试从 input_requirements 获取 (旧逻辑兼容)
        input_requirements = model_info.get('input_requirements', {})
        required_features = input_requirements.get('required_columns', input_requirements.get('features', []))
        
        # 决策: 使用哪个作为最终特征列表?
        # 如果经过预处理，data的列应该变成了处理后的列
        # 我们应该尽量匹配 model_features (训练时的列)
        
        final_features = model_features if model_features else required_features
        logger.info(f"Final features selected: {final_features}")
        
        if not final_features:
            # 如果都找不到，直接返回 data.values (可能报错)
            return data.values

        # 检查缺失列并填充0 (为了鲁棒性)
        missing_features = [f for f in final_features if f not in data.columns]
        if missing_features:
            logger.warning(f"Missing features in prepared data: {missing_features}. Filling with 0.")
            # 可能是特征选择删除了它们? 
            # 或者 OneHot 编码产生的列不匹配 (比如新数据缺少某个类别)?
            # 对于 OneHot，缺少的类别列应该补0
            for f in missing_features:
                data[f] = 0
        
        # 确保列顺序一致
        X = data[final_features].copy()
        
        # 处理可能残留的缺失值 (预处理后应该没有了，但为了安全)
        if X.isnull().any().any():
            X = X.fillna(0)
            
        return X.values
    
    def _inverse_transform_predictions(self, predictions: np.ndarray, model_id: int) -> np.ndarray:
        """
        反转预测结果的变换 (反归一化/反编码)
        """
        try:
            model_info = self.db_manager.get_model(model_id)
            if not model_info:
                return predictions
                
            # 获取目标列名
            input_requirements = model_info.get('input_requirements', {})
            target_columns = input_requirements.get('target', [])
            
            if not target_columns:
                logger.info(f"模型 {model_id} 未定义目标列，跳过反向变换")
                return predictions
                
            target_column = target_columns[0].strip()
            logger.info(f"模型 {model_id} 目标列: {target_column}, 准备反向变换")
                
            # 获取所有预处理组件
            components = self.db_manager.get_preprocessing_components(model_id)
            if not components:
                logger.info(f"模型 {model_id} 无预处理组件")
                return predictions
                
            # 筛选出作用于目标列的组件
            target_components = []
            
            def normalize_col(name):
                return str(name).strip().lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', '')
                
            target_column_norm = normalize_col(target_column)
            
            for comp in components:
                applied_cols = comp.get('applied_columns', [])
                # 确保 applied_cols 是列表
                if isinstance(applied_cols, str):
                    try:
                        applied_cols = json.loads(applied_cols)
                    except:
                        applied_cols = [applied_cols]
                
                # 使用归一化匹配
                for col in applied_cols:
                    if normalize_col(col) == target_column_norm:
                        target_components.append(comp)
                        logger.info(f"找到目标列组件: {comp.get('component_name')} ({comp.get('component_type')})")
                        break
            
            if not target_components:
                logger.info(f"未找到作用于目标列 {target_column} 的组件")
                return predictions

            # 按应用顺序的倒序执行反变换 (Scaler -> Encoder)
            original_predictions = predictions.copy()
            
            for comp in reversed(target_components):
                try:
                    file_path = comp.get('file_path')
                    
                    # 处理相对路径
                    if file_path and not os.path.isabs(file_path):
                        from backend.config import BASE_DIR
                        file_path = os.path.join(BASE_DIR, file_path)
                        
                    if not file_path or not os.path.exists(file_path):
                        logger.warning(f"组件文件不存在: {file_path}")
                        continue
                        
                    transformer = joblib.load(file_path)
                    c_type = comp['component_type']
                    
                    logger.info(f"正在应用反向变换: {comp.get('component_name')} ({c_type})")
                    
                    if c_type == 'scaler':
                        # Scaler通常期望 2D array
                        is_1d = original_predictions.ndim == 1
                        if is_1d:
                            original_predictions = original_predictions.reshape(-1, 1)
                            
                        original_predictions = transformer.inverse_transform(original_predictions)
                        
                        # 恢复为 1D array
                        if is_1d:
                            original_predictions = original_predictions.ravel()
                        
                    elif c_type == 'encoder':
                        # LabelEncoder
                        if isinstance(transformer, LabelEncoder):
                             # LabelEncoder期望整数
                             # 注意：如果是分类预测，predictions可能是概率或浮点数，需先转int
                             # 但通常 LabelEncoder 用于分类任务的 predict 结果 (即已经是 0, 1, 2)
                             original_predictions = transformer.inverse_transform(original_predictions.astype(int))
                             
                    logger.info(f"反向变换成功: {comp.get('component_name')}")
                             
                except Exception as e:
                    logger.warning(f"反转变换失败 {comp.get('component_name')}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            return original_predictions
        except Exception as e:
            logger.error(f"反向变换过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            return predictions
            logger.error(f"反转预测结果失败: {e}")
            return predictions

    def _save_prediction_record(
        self,
        model_id: int,
        input_data: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: Optional[List] = None
    ) -> int:
        """
        保存预测记录
        
        Args:
            model_id: 模型ID
            input_data: 输入数据
            predictions: 预测结果
            probabilities: 预测概率
            
        Returns:
            预测记录ID
        """
        # 获取原始预测结果
        original_predictions = self._inverse_transform_predictions(predictions, model_id)
        
        prediction_summary = {
            'input_data': input_data.to_dict('records'),
            'predictions': predictions.tolist(),
            'original_predictions': original_predictions.tolist(),
            'probabilities': probabilities,
            'n_samples': len(predictions),
            'created_at': datetime.now().isoformat()
        }
        
        # 保存为临时文件
        output_file = PREDICTIONS_DIR / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 将预测结果添加到输入数据
        result_df = input_data.copy()
        result_df['prediction'] = predictions
        result_df['prediction_original'] = original_predictions
        
        if probabilities is not None:
            # 如果有概率，也保存
            # 这里简化处理，只保存概率列表的字符串表示，或者扩展多列
            result_df['probabilities'] = [str(p) for p in probabilities]
            
        result_df.to_csv(output_file, index=False)
        
        prediction_id = self.db_manager.create_prediction(
            model_id=model_id,
            input_dataset_id=0,  # 0表示直接输入
            output_file_path=str(output_file),
            prediction_summary=prediction_summary
        )
        return prediction_id
    
    def get_prediction_history(
        self,
        model_id: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取预测历史
        
        Args:
            model_id: 模型ID,None表示获取所有
            limit: 返回数量限制
            
        Returns:
            预测历史列表
        """
        predictions = self.db_manager.get_predictions(model_id=model_id, limit=limit)
        
        # 处理历史记录，确保 'predictions' 字段是反缩放后的数据
        for pred in predictions:
            summary = pred.get('prediction_summary', {})
            if summary:
                # 如果存在 original_predictions，优先使用它作为展示的 predictions
                if 'original_predictions' in summary:
                    summary['predictions'] = summary['original_predictions']
                # 如果没有，检查是否存在 prediction_original (可能是文件保存时的字段名)
                # 但 summary 结构通常是我们在 _save_prediction_record 中定义的
                
                # 更新 summary
                pred['prediction_summary'] = summary
                
        return predictions
