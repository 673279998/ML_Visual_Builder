"""
预处理API路由
Preprocessing Routes
"""
from flask import Blueprint, request, jsonify
import logging
from backend.services.preprocessing_service import PreprocessingService
from backend.services.data_service import DataService

logger = logging.getLogger(__name__)

preprocessing_bp = Blueprint('preprocessing', __name__)
preprocessing_service = PreprocessingService()
data_service = DataService()


# 定义中间数据集的后缀
INTERMEDIATE_SUFFIXES = [
    '_missing_handled', 
    '_encoded', 
    '_scaled', 
    '_outliers_handled', 
    '_features_selected',
    '_processed',  # 统一后缀
    '_处理后的数据集'  # 兼容旧后缀
]

def _generate_processed_dataset_name(current_name):
    """生成处理后的数据集名称，避免后缀堆叠"""
    base_name = current_name
    # 循环去除所有已知后缀，直到没有后缀为止
    while True:
        found = False
        for suffix in INTERMEDIATE_SUFFIXES:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                found = True
                break
        if not found:
            break
    
    return f"{base_name}_processed"


def _save_components_to_db(dataset_id, new_dataset_id, info, component_type, workflow_id=None):
    """保存组件信息到dataset_encoders表"""
    if not info:
        return
        
    components = []
    if component_type == 'imputer':
        components = info.get('saved_imputers', [])
    elif component_type == 'encoder':
        components = info.get('saved_encoders', [])
    elif component_type == 'scaler':
        if info.get('saved_scalers'):
            components = info['saved_scalers']
        elif info.get('saved_scaler'):
            components = [info['saved_scaler']]
            
    for comp in components:
        # 构造摘要信息
        summary = {
            'type': component_type,
            'details': comp
        }
        
        # 构造名称
        if component_type == 'scaler':
            # 区分特征和目标scaler
            scaler_type = comp.get('type', 'features')
            name = f"{component_type}_{scaler_type}_{comp.get('method', 'unknown')}"
        elif component_type == 'encoder':
             name = f"{component_type}_{comp.get('column', 'col')}_{comp.get('method', 'unknown')}"
        else:
             name = f"{component_type}_{comp.get('method', 'unknown')}"
        
        # 确定应用的列
        cols = comp.get('columns', [])
        if not cols and comp.get('column'):
            cols = [comp.get('column')]

        # 保存到数据库
        # 注意：利用dataset_encoders表存储所有类型的组件
        try:
            # 确保数据可JSON序列化
            from backend.utils.json_utils import sanitize_for_json
            safe_column_mappings = sanitize_for_json({'columns': cols})
            safe_summary = sanitize_for_json(summary)
            
            data_service.db.create_encoder(
                name=name,
                source_dataset_id=dataset_id,
                encoded_dataset_id=new_dataset_id,
                file_path=comp['path'],
                column_mappings=safe_column_mappings, 
                encoding_summary=safe_summary,
                workflow_id=workflow_id
            )
            logger.info(f"已保存组件到数据库: {name}, 源数据集: {dataset_id}, 目标数据集: {new_dataset_id}, 工作流: {workflow_id}")
        except Exception as e:
            logger.error(f"保存组件到数据库失败: {e}")
            import traceback
            logger.error(traceback.format_exc())


def _cleanup_intermediate_dataset(dataset_id, data_service):
    """
    清理中间数据集
    如果数据集是中间产生的（根据名称后缀判断），则删除它
    """
    try:
        dataset_info = data_service.get_dataset_info(dataset_id)
        name = dataset_info['dataset']['name']
        
        # 检查是否匹配任何后缀
        is_intermediate = False
        for suffix in INTERMEDIATE_SUFFIXES:
            if name.endswith(suffix):
                # 特殊保护：不自动删除统一命名的数据集
                # 避免误删用户选定作为起点的已处理数据集
                # if suffix == '_processed' or suffix == '_处理后的数据集':
                #     continue
                    
                is_intermediate = True
                break
        
        if is_intermediate:
            logger.info(f"Cleaning up intermediate dataset: {name} (ID: {dataset_id})")
            data_service.delete_dataset(dataset_id)
            
    except Exception as e:
        logger.warning(f"Failed to cleanup dataset {dataset_id}: {str(e)}")


@preprocessing_bp.route('/api/preprocess/missing-values', methods=['POST'])
def handle_missing_values():
    """
    处理缺失值
    
    请求体:
    {
        "dataset_id": 1,
        "workflow_id": "wf-123", // 可选
        "strategy": "mean",  // mean, median, most_frequent, constant, drop
        "columns": ["age", "income"],  // 可选,null表示所有列
        "fill_value": null  // strategy为constant时必需
    }
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        workflow_id = data.get('workflow_id')
        strategy = data.get('strategy', 'mean')
        columns = data.get('columns')
        fill_value = data.get('fill_value')
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        # 加载数据集
        dataset = data_service.load_dataset(dataset_id)
        
        # 获取目标列和分类列
        db_columns = data_service.db.get_columns(dataset_id)
        target_columns = [col['column_name'] for col in db_columns if col.get('is_target')]
        categorical_columns = [col['column_name'] for col in db_columns if col.get('data_type') == 'categorical']
        
        # 处理缺失值
        processed_data, info = preprocessing_service.handle_missing_values(
            data=dataset['data'],
            strategy=strategy,
            columns=columns,
            fill_value=fill_value,
            dataset_id=dataset_id,
            workflow_id=workflow_id,
            target_columns=target_columns,
            categorical_columns=categorical_columns
        )
        
        # 生成新的数据集名称
        base_name = dataset['dataset']['name']
        # 移除可能的旧后缀
        for suffix in INTERMEDIATE_SUFFIXES:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        new_name = f"{base_name}_imputed"

        # 保存处理后的数据集
        new_dataset_id = data_service.save_dataset(
            data=processed_data,
            name=new_name,
            source_dataset_id=dataset_id
        )
        
        # 保存组件信息
        _save_components_to_db(dataset_id, new_dataset_id, info, 'imputer', workflow_id)
        
        # 清理中间数据集
        # _cleanup_intermediate_dataset(dataset_id, data_service)
        
        return jsonify({
            'success': True,
            'data': {
                'new_dataset_id': new_dataset_id,
                'info': info
            }
        })
        
    except Exception as e:
        logger.error(f"处理缺失值失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


from backend.utils.json_utils import sanitize_for_json

@preprocessing_bp.route('/api/preprocess/encode', methods=['POST'])
def encode_features():
    """
    自动编码特征
    
    请求体:
    {
        "dataset_id": 1,
        "workflow_id": "wf-123" // 可选
    }
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        workflow_id = data.get('workflow_id')
        target_column = data.get('target_column')
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        # 加载数据集
        dataset = data_service.load_dataset(dataset_id)
        
        # 获取列配置
        db_columns = data_service.db.get_columns(dataset_id)
        
        # 获取前端传入的编码配置
        encoding_configs = data.get('encoding_configs', [])
        
        # 合并配置
        final_columns = db_columns
        if encoding_configs:
            # 创建映射以便快速查找
            db_col_map = {col['column_name']: col for col in db_columns}
            
            for config in encoding_configs:
                col_name = config.get('column_name')
                if col_name and col_name in db_col_map:
                    # 更新DB中的配置（仅在内存中更新，用于本次操作）
                    db_col_map[col_name].update(config)
            
            final_columns = list(db_col_map.values())
        
        # 自动编码
        encoded_data, info = preprocessing_service.auto_encode_features(
            data=dataset['data'],
            column_configs=final_columns,
            dataset_id=dataset_id,
            workflow_id=workflow_id,
            target_column=target_column
        )
        
        # 立即清理 info 中的 numpy 类型，防止后续数据库操作或 JSON 响应失败
        from backend.utils.json_utils import sanitize_for_json
        info = sanitize_for_json(info)
        
        # 检查是否有编码错误
        if info.get('errors'):
            error_msg = "; ".join(info['errors'])
            return jsonify({
                'success': False,
                'error': f"特征编码失败: {error_msg}"
            }), 400
        
        # 确保encoded_data中没有numpy类型，转换为Python原生类型
        # 最简单的方法：将DataFrame转换为字典列表，然后让sanitize_for_json处理
        import pandas as pd
        records = encoded_data.to_dict(orient='records')
        sanitized_records = sanitize_for_json(records)
        encoded_data = pd.DataFrame(sanitized_records)
        
        # 生成新的数据集名称
        base_name = dataset['dataset']['name']
        # 移除可能的旧后缀
        for suffix in INTERMEDIATE_SUFFIXES:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        new_name = f"{base_name}_encoded"

        # 保存处理后的数据集
        new_dataset_id = data_service.save_dataset(
            data=encoded_data,
            name=new_name,
            source_dataset_id=dataset_id
        )
        
        # 保存组件信息
        _save_components_to_db(dataset_id, new_dataset_id, info, 'encoder', workflow_id)
        
        # 清理中间数据集
        # _cleanup_intermediate_dataset(dataset_id, data_service)
        
        return jsonify({
            'success': True,
            'data': {
                'new_dataset_id': new_dataset_id,
                'info': info
            }
        })
        
    except Exception as e:
        import traceback
        logger.error(f"自动编码失败: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        # 尝试打印 info 的结构以便调试
        try:
            if 'info' in locals():
                logger.error(f"Info object causing error (partial): {str(info)[:500]}")
        except:
            pass
            
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@preprocessing_bp.route('/api/preprocess/scale', methods=['POST'])
def scale_features():
    """
    数据缩放/标准化
    
    请求体:
    {
        "dataset_id": 1,
        "workflow_id": "wf-123", // 可选
        "method": "standard",  // standard, minmax, robust
        "columns": ["age", "income"]  // 可选,null表示所有数值列
    }
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        workflow_id = data.get('workflow_id')
        method = data.get('method', 'standard')
        columns = data.get('columns')
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        # 加载数据集
        dataset = data_service.load_dataset(dataset_id)
        
        # 获取目标列和分类列
        db_columns = data_service.db.get_columns(dataset_id)
        target_columns = [col['column_name'] for col in db_columns if col.get('is_target')]
        categorical_columns = [col['column_name'] for col in db_columns if col.get('data_type') == 'categorical']
        
        # 缩放数据
        scaled_data, info = preprocessing_service.scale_features(
            data=dataset['data'],
            method=method,
            columns=columns,
            dataset_id=dataset_id,
            workflow_id=workflow_id,
            target_columns=target_columns,
            categorical_columns=categorical_columns
        )
        
        # 生成新的数据集名称
        base_name = dataset['dataset']['name']
        # 移除可能的旧后缀
        for suffix in INTERMEDIATE_SUFFIXES:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        new_name = f"{base_name}_scaled"

        # 保存处理后的数据集
        new_dataset_id = data_service.save_dataset(
            data=scaled_data,
            name=new_name,
            source_dataset_id=dataset_id
        )
        
        # 保存组件信息
        _save_components_to_db(dataset_id, new_dataset_id, info, 'scaler', workflow_id)
        
        # 清理中间数据集
        # _cleanup_intermediate_dataset(dataset_id, data_service)
        
        return jsonify({
            'success': True,
            'data': {
                'new_dataset_id': new_dataset_id,
                'info': info
            }
        })
        
    except Exception as e:
        logger.error(f"数据缩放失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@preprocessing_bp.route('/api/preprocess/outliers/detect', methods=['POST'])
def detect_outliers():
    """
    检测异常值
    
    请求体:
    {
        "dataset_id": 1,
        "method": "iqr",  // iqr, zscore
        "columns": ["age", "income"],  // 可选
        "threshold": 1.5
    }
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        method = data.get('method', 'iqr')
        columns = data.get('columns')
        threshold = data.get('threshold', 1.5)
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        # 加载数据集
        dataset = data_service.load_dataset(dataset_id)
        
        # 获取分类列
        db_columns = data_service.db.get_columns(dataset_id)
        categorical_columns = [col['column_name'] for col in db_columns if col.get('data_type') == 'categorical']
        
        # 检测异常值
        _, outliers_info = preprocessing_service.detect_outliers(
            data=dataset['data'],
            method=method,
            columns=columns,
            threshold=threshold,
            categorical_columns=categorical_columns
        )
        
        return jsonify({
            'success': True,
            'data': outliers_info
        })
        
    except Exception as e:
        logger.error(f"检测异常值失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@preprocessing_bp.route('/api/preprocess/outliers/handle', methods=['POST'])
def handle_outliers():
    """
    处理异常值
    
    请求体:
    {
        "dataset_id": 1,
        "method": "clip",  // clip, remove
        "columns": ["age", "income"],
        "detection_method": "iqr",
        "threshold": 1.5
    }
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        method = data.get('method', 'clip')
        columns = data.get('columns')
        detection_method = data.get('detection_method', 'iqr')
        threshold = data.get('threshold', 1.5)
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        # 加载数据集
        dataset = data_service.load_dataset(dataset_id)
        
        # 获取分类列
        db_columns = data_service.db.get_columns(dataset_id)
        categorical_columns = [col['column_name'] for col in db_columns if col.get('data_type') == 'categorical']
        
        # 处理异常值
        processed_data, info = preprocessing_service.handle_outliers(
            data=dataset['data'],
            method=detection_method,
            action=method,
            columns=columns,
            threshold=threshold,
            categorical_columns=categorical_columns
        )
        
        # 保存处理后的数据集
        new_dataset_id = data_service.save_dataset(
            data=processed_data,
            name=_generate_processed_dataset_name(dataset['dataset']['name']),
            source_dataset_id=dataset_id
        )
        
        # 清理中间数据集
        # _cleanup_intermediate_dataset(dataset_id, data_service)
        
        return jsonify({
            'success': True,
            'data': {
                'new_dataset_id': new_dataset_id,
                'info': info
            }
        })
        
    except Exception as e:
        logger.error(f"处理异常值失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@preprocessing_bp.route('/api/preprocess/features/select', methods=['POST'])
def select_features():
    """
    特征选择
    
    请求体:
    {
        "dataset_id": 1,
        "method": "variance",  // variance, correlation, importance
        "n_features": 10,
        "threshold": 0.01
    }
    """
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        method = data.get('method', 'variance')
        n_features = data.get('n_features')
        threshold = data.get('threshold', 0.01)
        target_column = data.get('target_column')
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        # 加载数据集
        dataset = data_service.load_dataset(dataset_id)

        columns_config = data_service.db.get_columns(dataset_id)
        categorical_columns = [col['column_name'] for col in columns_config if col.get('data_type') == 'categorical']

        # 如果未指定目标列，尝试从数据集配置中获取
        if not target_column:
            target_columns_list = [col['column_name'] for col in columns_config if col.get('is_target')]
            if target_columns_list:
                target_column = target_columns_list[0]
        
        if not target_column:
            return jsonify({
                'success': False,
                'error': '未指定目标变量，且数据集中未定义目标变量'
            }), 400
        
        # 特征选择
        selected_data, info = preprocessing_service.select_features(
            data=dataset['data'],
            target_column=target_column,
            method=method,
            k=n_features if n_features else 10,
            threshold=threshold,
            categorical_columns=categorical_columns
        )
        
        # 保存处理后的数据集
        new_dataset_id = data_service.save_dataset(
            data=selected_data,
            name=_generate_processed_dataset_name(dataset['dataset']['name']),
            source_dataset_id=dataset_id
        )
        
        # 清理中间数据集
        # _cleanup_intermediate_dataset(dataset_id, data_service)
        
        return jsonify({
            'success': True,
            'data': {
                'new_dataset_id': new_dataset_id,
                'info': info
            }
        })
        
    except Exception as e:
        logger.error(f"特征选择失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@preprocessing_bp.route('/api/preprocess/history', methods=['GET'])
def get_preprocessing_history():
    """获取预处理历史"""
    try:
        history = preprocessing_service.preprocessing_history
        
        return jsonify({
            'success': True,
            'data': history
        })
        
    except Exception as e:
        logger.error(f"获取预处理历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
