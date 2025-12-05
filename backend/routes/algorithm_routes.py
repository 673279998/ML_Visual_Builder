"""
算法路由 - 提供算法列表和配置信息
"""
from flask import Blueprint, jsonify, request
from backend.hyperparameter.hyperparameter_registry import HyperparameterRegistry

algorithm_bp = Blueprint('algorithm', __name__)
registry = HyperparameterRegistry()

@algorithm_bp.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """获取所有算法列表"""
    try:
        algorithms = registry.get_all_algorithms()
        return jsonify({
            'success': True,
            'data': algorithms
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@algorithm_bp.route('/api/algorithms/<algorithm_type>', methods=['GET'])
def get_algorithms_by_type(algorithm_type):
    """按类型获取算法列表"""
    try:
        algorithms = registry.get_algorithms_by_type(algorithm_type)
        return jsonify({
            'success': True,
            'data': algorithms
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@algorithm_bp.route('/api/algorithms/<algorithm_name>/hyperparameters', methods=['GET'])
def get_hyperparameters(algorithm_name):
    """获取算法的超参数定义"""
    try:
        hyperparams = registry.get_hyperparameters(algorithm_name)
        if not hyperparams:
            return jsonify({
                'success': False,
                'error': '算法不存在'
            }), 404
        return jsonify({
            'success': True,
            'data': hyperparams
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
