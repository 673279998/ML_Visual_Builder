"""
超参数调优API路由
Hyperparameter Tuning Routes
"""
from flask import Blueprint, request, jsonify
import logging
import uuid
import threading
from backend.services.hyperparameter_tuning_service import HyperparameterTuner, HyperparameterRegistry
from backend.services.data_service import DataService
from backend.utils.progress_tracker import progress_tracker

logger = logging.getLogger(__name__)

hyperparameter_bp = Blueprint('hyperparameter', __name__)
tuner = HyperparameterTuner()
data_service = DataService()


@hyperparameter_bp.route('/api/hyperparameter/tune', methods=['POST'])
def tune_hyperparameters():
    """
    异步执行超参数调优（支持进度监控）
    
    请求体:
    {
        "dataset_id": 1,
        "algorithm_name": "random_forest_classifier",
        "target_columns": ["target"],
        "tuning_method": "grid_search",  // grid_search, random_search, bayesian
        "param_grid": {...},  // 可选,不传则使用推荐参数
        "cv": 5,
        "scoring": "accuracy",
        "n_iter": 100  // 随机搜索和贝叶斯优化使用
    }
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        dataset_id = data.get('dataset_id')
        algorithm_name = data.get('algorithm_name')
        target_columns = data.get('target_columns', [])
        tuning_method = data.get('tuning_method', 'grid_search')
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: dataset_id'
            }), 400
        
        if not algorithm_name:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: algorithm_name'
            }), 400
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建进度追踪任务
        progress_tracker.create_task(task_id, 'tuning', total_steps=3)
        
        # 异步执行调优
        def tuning_task(target_columns_arg):
            try:
                progress_tracker.update_progress(task_id, current_step=1, message='加载数据...')
                
                # 加载数据集
                dataset = data_service.load_dataset(dataset_id)
                df = dataset['data']
                
                # 如果未指定目标列，尝试从数据集配置中获取
                if not target_columns_arg:
                    columns_config = data_service.db.get_columns(dataset_id)
                    target_columns_arg = [col['column_name'] for col in columns_config if col['is_target']]
                
                # 分离特征和目标
                if target_columns_arg:
                    y = df[target_columns_arg].values
                    X = df.drop(columns=target_columns_arg).values
                    # 如果只有一个目标列,展平
                    if y.shape[1] == 1:
                        y = y.ravel()
                else:
                    X = df.values
                    y = None
                
                # 获取参数网格/分布
                param_grid = data.get('param_grid')
                
                # 检查传入的param_grid是否有效（是否包含可搜索的空间）
                if param_grid:
                    is_valid_space = False
                    for val in param_grid.values():
                        if isinstance(val, list) and len(val) > 1:
                            is_valid_space = True
                            break
                    
                    if not is_valid_space:
                        logger.info("传入的参数网格仅包含固定值，将使用默认推荐参数进行调优")
                        param_grid = None

                if not param_grid:
                    if tuning_method in ['grid_search', 'grid']:
                        param_grid = HyperparameterRegistry.get_param_grid(algorithm_name)
                    else:
                        param_grid = HyperparameterRegistry.get_param_distributions(algorithm_name)
                
                if not param_grid:
                    raise ValueError(f'算法 {algorithm_name} 没有可用的参数配置')
                
                # 获取其他参数
                cv = data.get('cv', 5)
                scoring = data.get('scoring')
                n_iter = data.get('n_iter', 100)
                
                progress_tracker.update_progress(task_id, current_step=2, 
                    message=f'执行{tuning_method}调优...')
                
                # 执行调优
                if tuning_method in ['grid_search', 'grid']:
                    result = tuner.grid_search(
                        algorithm_name=algorithm_name,
                        X_train=X,
                        y_train=y,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=scoring
                    )
                elif tuning_method in ['random_search', 'random']:
                    result = tuner.random_search(
                        algorithm_name=algorithm_name,
                        X_train=X,
                        y_train=y,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv,
                        scoring=scoring
                    )
                elif tuning_method in ['bayesian', 'bayesian_optimization']:
                    result = tuner.bayesian_optimization(
                        algorithm_name=algorithm_name,
                        X_train=X,
                        y_train=y,
                        param_space=param_grid,
                        n_iter=n_iter,
                        cv=cv,
                        scoring=scoring
                    )
                else:
                    raise ValueError(f'不支持的调优方法: {tuning_method}')
                
                progress_tracker.update_progress(task_id, current_step=3, message='调优完成')
                progress_tracker.complete_task(task_id, '超参数调优完成', result)
                
            except Exception as e:
                logger.error(f"超参数调优失败: {str(e)}")
                progress_tracker.fail_task(task_id, str(e))
        
        # 启动后台线程
        thread = threading.Thread(target=tuning_task, args=(target_columns,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '调优任务已启动'
        }), 200
        
    except Exception as e:
        logger.error(f"启动调优任务失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@hyperparameter_bp.route('/api/hyperparameter/tune/progress/<task_id>', methods=['GET'])
def get_tuning_progress(task_id):
    """获取调优任务进度"""
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
        })
        
    except Exception as e:
        logger.error(f"获取调优进度失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@hyperparameter_bp.route('/api/hyperparameter/param-grid/<algorithm_name>', methods=['GET'])
def get_param_grid(algorithm_name):
    """获取算法的推荐参数网格"""
    try:
        param_grid = HyperparameterRegistry.get_param_grid(algorithm_name)
        
        if not param_grid:
            return jsonify({
                'success': False,
                'error': f'算法 {algorithm_name} 没有推荐参数网格'
            }), 404
        
        return jsonify({
            'success': True,
            'data': param_grid
        })
        
    except Exception as e:
        logger.error(f"获取参数网格失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@hyperparameter_bp.route('/api/hyperparameter/param-distributions/<algorithm_name>', methods=['GET'])
def get_param_distributions(algorithm_name):
    """获取算法的推荐参数分布"""
    try:
        param_dist = HyperparameterRegistry.get_param_distributions(algorithm_name)
        
        if not param_dist:
            return jsonify({
                'success': False,
                'error': f'算法 {algorithm_name} 没有推荐参数分布'
            }), 404
        
        # 将scipy分布转换为可序列化的格式
        serializable_dist = {}
        for key, value in param_dist.items():
            if hasattr(value, '__class__'):
                serializable_dist[key] = {
                    'type': value.__class__.__name__,
                    'params': str(value)
                }
            else:
                serializable_dist[key] = value
        
        return jsonify({
            'success': True,
            'data': serializable_dist
        })
        
    except Exception as e:
        logger.error(f"获取参数分布失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@hyperparameter_bp.route('/api/hyperparameter/history', methods=['GET'])
def get_tuning_history():
    """获取调优历史"""
    try:
        summary = tuner.get_tuning_summary()
        
        return jsonify({
            'success': True,
            'data': summary
        })
        
    except Exception as e:
        logger.error(f"获取调优历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
