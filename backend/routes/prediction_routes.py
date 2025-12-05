"""
预测API路由
Prediction Routes
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import logging
from backend.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)
prediction_service = PredictionService()


@prediction_bp.route('/api/predict', methods=['POST'])
def predict():
    """
    单次预测
    
    请求体:
    {
        "model_id": 1,
        "input_data": [...],  # 数据数组或对象数组
        "return_probabilities": true
    }
    """
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        input_data = data.get('input_data')
        return_probabilities = data.get('return_probabilities', False)
        
        if not model_id or not input_data:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: model_id 或 input_data'
            }), 400
        
        # 转换为DataFrame
        df = pd.DataFrame(input_data)
        
        # 进行预测
        result = prediction_service.predict(
            model_id=model_id,
            input_data=df,
            return_probabilities=return_probabilities
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/api/predict/batch', methods=['POST'])
def batch_predict():
    """
    批量预测
    
    请求体:
    {
        "model_id": 1,
        "dataset_id": 2
    }
    """
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        dataset_id = data.get('dataset_id')
        
        if not model_id or not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: model_id 或 dataset_id'
            }), 400
        
        # 获取数据集
        from backend.services.data_service import DataService
        data_service = DataService()
        dataset = data_service.load_dataset(dataset_id)
        
        # 进行预测
        result = prediction_service.predict(
            model_id=model_id,
            input_data=dataset['data'],
            return_probabilities=True
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/api/predictions', methods=['GET'])
def get_predictions():
    """
    获取预测历史列表
    
    查询参数:
    - model_id: 模型ID (可选)
    - limit: 返回数量限制 (默认50)
    """
    try:
        model_id = request.args.get('model_id', type=int)
        limit = request.args.get('limit', default=50, type=int)
        
        predictions = prediction_service.get_prediction_history(
            model_id=model_id,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'data': predictions
        })
        
    except Exception as e:
        logger.error(f"获取预测历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/api/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    """获取预测记录详情"""
    try:
        from backend.database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        prediction = db_manager.get_prediction(prediction_id)
        
        if not prediction:
            return jsonify({
                'success': False,
                'error': f'预测记录不存在: {prediction_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': prediction
        })
        
    except Exception as e:
        logger.error(f"获取预测详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/api/predictions/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """删除预测记录"""
    try:
        from backend.database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        success = db_manager.delete_prediction(prediction_id)
        
        if not success:
            return jsonify({
                'success': False,
                'error': f'预测记录不存在: {prediction_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'message': '预测记录删除成功'
        })
        
    except Exception as e:
        logger.error(f"删除预测记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/api/models/<int:model_id>/predictions', methods=['GET'])
def get_model_predictions(model_id):
    """获取指定模型的所有预测记录"""
    try:
        from backend.database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        predictions = db_manager.get_model_predictions(model_id)
        
        return jsonify({
            'success': True,
            'data': predictions
        })
        
    except Exception as e:
        logger.error(f"获取模型预测记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
