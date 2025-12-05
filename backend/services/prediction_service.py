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
        """应用预处理组件"""
        df = data.copy()
        
        # 自动处理日期列
        preprocessor = PreprocessingService()
        df = preprocessor._handle_datetime_columns(df)
        
        # 按顺序应用组件
        # 组件列表应当已经按照 Imputer -> Encoder -> Scaler 的顺序排列
        for comp in components:
            c_type = comp['component_type']
            c_path = comp['file_path']
            c_cols = comp['applied_columns']
            c_config = comp['configuration']
            
            if not c_path or not os.path.exists(c_path):
                error_msg = f"预处理组件文件不存在: {c_path} (组件: {comp.get('component_name')})"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            try:
                transformer = joblib.load(c_path)
                
                # 验证列是否存在
                missing_cols = [c for c in c_cols if c not in df.columns]
                # 对于Encoder，如果列已经在之前的步骤被处理（比如被Imputer处理过），可能还在
                # 但如果被OneHot处理过，列名会变。所以顺序很重要。
                if missing_cols:
                    # 可能是之前的步骤改变了列名，或者输入数据缺失
                    # 如果是特征选择，不需要列存在，只需要过滤
                    if c_type != 'feature_selector':
                         # 检查是否是OneHot之后的情况? 不，Encoder处理原始列。
                         logger.warning(f"组件 {comp['component_name']} 缺少输入列: {missing_cols}")
                         continue

                if c_type == 'imputer':
                    # Imputer通常返回numpy array，我们需要保持DataFrame格式
                    # Sklearn Imputer transform
                    data_part = df[c_cols]
                    transformed = transformer.transform(data_part)
                    
                    # 如果是SimpleImputer，transformed shape和c_cols一致
                    if isinstance(transformed, pd.DataFrame):
                        df[c_cols] = transformed
                    else:
                        df[c_cols] = transformed
                        
                elif c_type == 'scaler':
                    # Scaler transform
                    data_part = df[c_cols]
                    transformed = transformer.transform(data_part)
                    if isinstance(transformed, pd.DataFrame):
                        df[c_cols] = transformed
                    else:
                        df[c_cols] = transformed
                        
                elif c_type == 'encoder':
                    # 处理编码器 (LabelEncoder, OneHotEncoder, etc.)
                    for col in c_cols:
                        if col not in df.columns:
                            continue
                            
                        series = df[col]
                        transformed = transformer.transform(series)
                        
                        if hasattr(transformed, 'toarray'): # 处理稀疏矩阵
                            transformed = transformed.toarray()
                            
                        if isinstance(transformed, pd.DataFrame):
                            # 如果返回的是DataFrame (如 category_encoders)
                            df = df.drop(columns=[col])
                            df = pd.concat([df, transformed], axis=1)
                        elif isinstance(transformed, np.ndarray):
                            if transformed.ndim == 1:
                                # LabelEncoder or similar: 替换原列
                                df[col] = transformed
                            else:
                                # OneHotEncoder (sklearn): 返回2D数组
                                # 尝试获取列名
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
                             # Fallback
                             df[col] = transformed

            except Exception as e:
                logger.error(f"应用预处理组件 {comp['component_name']} 失败: {str(e)}")
                # 继续尝试下一个组件
                continue
                
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
        # 1. 获取预处理组件
        model_id = model_info['id']
        components = self.db_manager.get_preprocessing_components(model_id)
        
        # 2. 应用预处理
        if components:
            data = self._apply_preprocessing(data, components)
        
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
        
        if not final_features:
            # 如果都找不到，直接返回 data.values (可能报错)
            return data.values

        # 检查缺失列并填充0 (为了鲁棒性)
        missing_features = [f for f in final_features if f not in data.columns]
        if missing_features:
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
                return predictions
                
            target_column = target_columns[0]
                
            # 获取所有预处理组件
            components = self.db_manager.get_preprocessing_components(model_id)
            if not components:
                return predictions
                
            # 筛选出作用于目标列的组件
            target_components = []
            for comp in components:
                if target_column in comp.get('applied_columns', []):
                    target_components.append(comp)
            
            # 按应用顺序的倒序执行反变换 (Scaler -> Encoder)
            original_predictions = predictions.copy()
            
            for comp in reversed(target_components):
                try:
                    if not comp.get('file_path') or not os.path.exists(comp['file_path']):
                        continue
                        
                    transformer = joblib.load(comp['file_path'])
                    c_type = comp['component_type']
                    
                    if c_type == 'scaler':
                        # Scaler通常期望 2D array
                        if original_predictions.ndim == 1:
                            original_predictions = original_predictions.reshape(-1, 1)
                        original_predictions = transformer.inverse_transform(original_predictions)
                        # 恢复为 1D array
                        original_predictions = original_predictions.ravel()
                        
                    elif c_type == 'encoder':
                        # LabelEncoder
                        if isinstance(transformer, LabelEncoder):
                             # LabelEncoder期望整数
                             original_predictions = transformer.inverse_transform(original_predictions.astype(int))
                except Exception as e:
                    logger.warning(f"反转变换失败 {comp['component_name']}: {e}")
                    
            return original_predictions
        except Exception as e:
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
        return predictions
