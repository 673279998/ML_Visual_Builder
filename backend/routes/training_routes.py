"""
训练相关API路由
"""
from flask import Blueprint, request, jsonify
from backend.services.training_service import TrainingService
from backend.database.db_manager import DatabaseManager
from backend.utils.progress_tracker import progress_tracker
import uuid
import threading

training_bp = Blueprint('training', __name__)
training_service = TrainingService()
db = DatabaseManager()


@training_bp.route('/api/train', methods=['POST'])
def train_model():
    """
    训练模型
    
    Request JSON:
    {
        "dataset_id": 1,
        "algorithm_name": "logistic_regression",
        "target_columns": ["target"],
        "feature_columns": ["feature1", "feature2"],  // 可选,不传则使用所有非目标列
        "test_size": 0.2,  // 可选,默认0.2
        "hyperparameters": {}  // 可选
    }
    """
    try:
        data = request.json
        
        # 验证必需参数
        if not data.get('dataset_id'):
            return jsonify({'error': '缺少dataset_id参数'}), 400
        if not data.get('algorithm_name'):
            return jsonify({'error': '缺少algorithm_name参数'}), 400
        
        # 训练模型
        result = training_service.train_model(
            dataset_id=data['dataset_id'],
            algorithm_name=data['algorithm_name'],
            target_columns=data.get('target_columns'),
            feature_columns=data.get('feature_columns'),
            test_size=data.get('test_size', 0.2),
            hyperparameters=data.get('hyperparameters'),
            preprocessing_config=data.get('preprocessing_config'),
            preprocessing_components=data.get('preprocessing_components')
        )
        
        return jsonify({
            'success': True,
            'message': '模型训练成功',
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'训练失败: {str(e)}'
        }), 500


@training_bp.route('/api/models', methods=['GET'])
def get_models():
    """
    获取所有模型列表
    """
    try:
        models = db.get_all_models()
        return jsonify({
            'success': True,
            'data': models
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/api/models/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """
    获取指定模型详情
    """
    try:
        model = db.get_model(model_id)
        if not model:
            return jsonify({
                'success': False,
                'error': '模型不存在'
            }), 404
        
        return jsonify({
            'success': True,
            'data': model
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/api/models/<int:model_id>/results', methods=['GET'])
def get_model_results(model_id):
    """
    获取模型的完整训练结果(包含可视化数据)
    """
    try:
        model = db.get_model(model_id)
        if not model:
            return jsonify({
                'success': False,
                'error': '模型不存在'
            }), 404
        
        # 返回性能指标、特征重要性和完整结果(包含所有可视化数据)
        response_data = {
            'model_id': model_id,
            'algorithm_name': model['algorithm_name'],
            'algorithm_type': model['algorithm_type'],
            'performance_metrics': model['performance_metrics'],
            'feature_importance': model.get('feature_importance'),
            'dataset_schema': model.get('dataset_schema'),
            'actual_hyperparameters': model.get('actual_hyperparameters'),  # 实际使用的超参数
            'complete_results': model.get('complete_results')  # 完整的训练结果(包含可视化)
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500










@training_bp.route('/api/train/async', methods=['POST'])
def train_model_async():
    """
    异步训练模型（支持进度监控）
    
    Request JSON:
    {
        "dataset_id": 1,
        "algorithm_name": "logistic_regression",
        "target_columns": ["target"],
        "feature_columns": ["feature1", "feature2"],  // 可选
        "test_size": 0.2,  // 可选
        "hyperparameters": {}  // 可选
    }
    
    Response:
    {
        "success": true,
        "task_id": "uuid-string",
        "message": "训练任务已启动"
    }
    """
    try:
        data = request.json
        
        # 验证必需参数
        if not data.get('dataset_id'):
            return jsonify({'error': '缺少dataset_id参数'}), 400
        if not data.get('algorithm_name'):
            return jsonify({'error': '缺少algorithm_name参数'}), 400
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建进度追踪任务
        progress_tracker.create_task(task_id, 'training', total_steps=5)
        
        # 异步执行训练
        def train_task():
            try:
                progress_tracker.update_progress(task_id, current_step=1, message='准备数据...')
                
                result = training_service.train_model(
                    dataset_id=data['dataset_id'],
                    algorithm_name=data['algorithm_name'],
                    target_columns=data.get('target_columns'),
                    feature_columns=data.get('feature_columns'),
                    test_size=data.get('test_size', 0.2),
                    hyperparameters=data.get('hyperparameters'),
                    preprocessing_config=data.get('preprocessing_config'),
                    preprocessing_components=data.get('preprocessing_components'),
                    task_id=task_id  # 传递task_id用于进度更新
                )
                
                progress_tracker.complete_task(task_id, '模型训练完成', result)
                
            except Exception as e:
                progress_tracker.fail_task(task_id, str(e))
        
        # 启动后台线程
        thread = threading.Thread(target=train_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '训练任务已启动'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'启动训练任务失败: {str(e)}'
        }), 500


@training_bp.route('/api/train/progress/<task_id>', methods=['GET'])
def get_training_progress(task_id):
    """
    获取训练进度
    
    Response:
    {
        "success": true,
        "data": {
            "task_id": "uuid-string",
            "task_type": "training",
            "status": "running",  // running, completed, failed, cancelled
            "progress": 50,  // 0-100
            "current_step": 2,
            "total_steps": 5,
            "message": "正在训练模型...",
            "start_time": "2025-12-01T10:00:00",
            "end_time": null,
            "error": null,
            "metadata": {}
        }
    }
    """
    try:
        progress = progress_tracker.get_progress(task_id)
        
        if progress is None:
            return jsonify({
                'success': False,
                'error': '任务不存在'
            }), 404
        
        return jsonify({
            'success': True,
            'data': progress
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/api/train/tasks', methods=['GET'])
def get_training_tasks():
    """
    获取所有训练任务列表
    
    Query Parameters:
        status: 任务状态过滤 (running, completed, failed, cancelled)
    """
    try:
        tasks = progress_tracker.get_all_tasks('training')
        
        # 根据状态过滤
        status_filter = request.args.get('status')
        if status_filter:
            tasks = [t for t in tasks if t['status'] == status_filter]
        
        return jsonify({
            'success': True,
            'data': tasks
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/api/train/cancel/<task_id>', methods=['POST'])
def cancel_training_task(task_id):
    """
    取消训练任务
    """
    try:
        progress = progress_tracker.get_progress(task_id)
        
        if progress is None:
            return jsonify({
                'success': False,
                'error': '任务不存在'
            }), 404
        
        if progress['status'] != 'running':
            return jsonify({
                'success': False,
                'error': '只能取消正在运行的任务'
            }), 400
        
        progress_tracker.cancel_task(task_id)
        
        return jsonify({
            'success': True,
            'message': '任务已取消'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
