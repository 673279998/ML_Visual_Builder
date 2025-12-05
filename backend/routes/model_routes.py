"""
模型管理相关路由
"""
from flask import Blueprint, request, jsonify
from backend.algorithms.algorithm_factory import AlgorithmFactory
from backend.database.db_manager import DatabaseManager
from backend.utils.json_utils import sanitize_for_json

model_bp = Blueprint('model', __name__, url_prefix='/api/models')
db = DatabaseManager()


@model_bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """获取所有可用算法列表"""
    try:
        algorithms = AlgorithmFactory.get_algorithm_list()
        return jsonify({
            'success': True,
            'data': algorithms
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/algorithms/<algorithm_name>', methods=['GET'])
def get_algorithm_info(algorithm_name):
    """获取算法详细信息"""
    try:
        info = AlgorithmFactory.get_algorithm_info(algorithm_name)
        return jsonify({
            'success': True,
            'data': info
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/algorithms/type/<algorithm_type>', methods=['GET'])
def get_algorithms_by_type(algorithm_type):
    """根据类型获取算法列表"""
    try:
        algorithms = AlgorithmFactory.get_algorithms_by_type(algorithm_type)
        return jsonify({
            'success': True,
            'data': algorithms
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/', methods=['GET'])
def get_models():
    """获取所有模型列表"""
    try:
        models = db.get_all_models()
        return jsonify({
            'success': True,
            'data': sanitize_for_json(models)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """获取模型详细信息"""
    try:
        model = db.get_model(model_id)
        if not model:
            return jsonify({'error': '模型不存在'}), 404
        
        # 获取训练结果
        result = db.get_training_result(model_id)
        if result:
            model['training_result'] = result
        
        # 获取预处理组件
        components = db.get_preprocessing_components(model_id)
        if components:
            model['preprocessing_components'] = components
        
        return jsonify({
            'success': True,
            'data': sanitize_for_json(model)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/<int:model_id>/results', methods=['GET'])
def get_model_results(model_id):
    """获取模型训练结果"""
    try:
        result = db.get_training_result(model_id)
        if not result:
            return jsonify({'error': '训练结果不存在'}), 404
            
        return jsonify({
            'success': True,
            'data': sanitize_for_json(result)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    """删除模型"""
    try:
        success = db.delete_model(model_id)
        if not success:
            return jsonify({'error': '模型不存在'}), 404
            
        return jsonify({
            'success': True,
            'message': '模型删除成功'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@model_bp.route('/batch', methods=['POST'])
def delete_models_batch():
    """批量删除模型"""
    try:
        data = request.get_json()
        if not data or 'model_ids' not in data:
            return jsonify({'error': '未提供model_ids'}), 400
            
        model_ids = data['model_ids']
        if not isinstance(model_ids, list):
            return jsonify({'error': 'model_ids必须是列表'}), 400
            
        deleted_count = 0
        errors = []
        
        for model_id in model_ids:
            try:
                success = db.delete_model(model_id)
                if success:
                    deleted_count += 1
                else:
                    errors.append(f"模型 {model_id} 不存在或删除失败")
            except Exception as e:
                errors.append(f"删除模型 {model_id} 失败: {str(e)}")
                
        return jsonify({
            'success': True,
            'message': f'成功删除 {deleted_count} 个模型',
            'deleted_count': deleted_count,
            'errors': errors
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
