"""
工作流API路由
Workflow Routes
"""
from flask import Blueprint, request, jsonify
import logging
from backend.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

workflow_bp = Blueprint('workflow', __name__)
db_manager = DatabaseManager()


@workflow_bp.route('/api/workflows', methods=['GET'])
def get_workflows():
    """获取所有工作流列表"""
    try:
        workflows = db_manager.get_all_workflows()
        
        return jsonify({
            'success': True,
            'data': workflows
        })
        
    except Exception as e:
        logger.error(f"获取工作流列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/api/workflows/<int:workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """获取工作流详情"""
    try:
        workflow = db_manager.get_workflow(workflow_id)
        
        if not workflow:
            return jsonify({
                'success': False,
                'error': f'工作流不存在: {workflow_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': workflow
        })
        
    except Exception as e:
        logger.error(f"获取工作流详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/api/workflows', methods=['POST'])
def create_workflow():
    """
    创建工作流
    
    请求体:
    {
        "name": "分类工作流",
        "workflow_type": "classification",
        "description": "鸢尾花分类",
        "configuration": {
            "nodes": [...],
            "connections": [...]
        }
    }
    """
    try:
        data = request.get_json()
        name = data.get('name')
        workflow_type = data.get('workflow_type')
        description = data.get('description')
        configuration = data.get('configuration')
        
        if not name or not workflow_type or not configuration:
            return jsonify({
                'success': False,
                'error': '缺少必需参数: name, workflow_type, configuration'
            }), 400
        
        # 创建工作流
        workflow_id = db_manager.create_workflow(
            name=name,
            workflow_type=workflow_type,
            description=description,
            configuration=configuration
        )
        
        return jsonify({
            'success': True,
            'data': {
                'workflow_id': workflow_id,
                'message': '工作流创建成功'
            }
        })
        
    except Exception as e:
        logger.error(f"创建工作流失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/api/workflows/<int:workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    """
    更新工作流
    
    请求体:
    {
        "name": "新名称",
        "description": "新描述",
        "configuration": {...}
    }
    """
    try:
        data = request.get_json()
        
        # 检查工作流是否存在
        workflow = db_manager.get_workflow(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': f'工作流不存在: {workflow_id}'
            }), 404
        
        # 更新工作流
        update_fields = {}
        if 'name' in data:
            update_fields['name'] = data['name']
        if 'description' in data:
            update_fields['description'] = data['description']
        if 'configuration' in data:
            update_fields['configuration'] = data['configuration']
        
        db_manager.update_workflow(workflow_id, **update_fields)
        
        return jsonify({
            'success': True,
            'data': {
                'message': '工作流更新成功'
            }
        })
        
    except Exception as e:
        logger.error(f"更新工作流失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/api/workflows/<int:workflow_id>', methods=['DELETE'])
def delete_workflow(workflow_id):
    """删除工作流"""
    try:
        # 检查工作流是否存在
        workflow = db_manager.get_workflow(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': f'工作流不存在: {workflow_id}'
            }), 404
        
        # 删除工作流
        db_manager.execute_update(
            'DELETE FROM workflows WHERE id = ?',
            (workflow_id,)
        )
        
        return jsonify({
            'success': True,
            'data': {
                'message': '工作流删除成功'
            }
        })
        
    except Exception as e:
        logger.error(f"删除工作流失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/api/workflows/batch', methods=['POST'])
def batch_delete_workflows():
    """批量删除工作流"""
    try:
        data = request.get_json()
        workflow_ids = data.get('workflow_ids', [])
        
        if not workflow_ids:
            return jsonify({
                'success': False,
                'error': '未提供workflow_ids'
            }), 400
            
        results = {
            'success': [],
            'failed': []
        }
        
        for workflow_id in workflow_ids:
            try:
                # 检查工作流是否存在
                workflow = db_manager.get_workflow(workflow_id)
                if not workflow:
                    results['failed'].append({'id': workflow_id, 'error': 'Workflow not found'})
                    continue
                
                # 删除工作流
                db_manager.execute_update(
                    'DELETE FROM workflows WHERE id = ?',
                    (workflow_id,)
                )
                results['success'].append(workflow_id)
            except Exception as e:
                results['failed'].append({'id': workflow_id, 'error': str(e)})
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        logger.error(f"批量删除工作流失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

