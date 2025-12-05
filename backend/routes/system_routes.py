"""
系统管理相关路由
System Management Routes
"""
from flask import Blueprint, jsonify
from backend.database.db_manager import DatabaseManager
from backend.services.data_service import DataService
from backend.services.cleanup_service import CleanupService
import logging

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)

# 初始化服务
# 注意: 这里实例化可能会在循环导入时出问题，如果出现，需要移到函数内部
db_manager = DatabaseManager()
data_service = DataService()
cleanup_service = CleanupService(db_manager, data_service)

@system_bp.route('/api/system/cleanup', methods=['POST'])
def cleanup_system():
    """
    清理系统中的垃圾文件
    包括:
    1. 无关联的处理后数据集
    2. 无关联的模型文件
    3. 无关联的预测结果
    4. 无关联的预处理组件(Encoder/Preprocessor)
    """
    try:
        results = cleanup_service.cleanup_all()
        
        logger.info(f"系统清理完成: {results}")
        
        return jsonify({
            'success': True,
            'message': '系统清理完成',
            'data': results
        })
        
    except Exception as e:
        logger.error(f"系统清理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
