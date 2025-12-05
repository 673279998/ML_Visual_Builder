"""
数据管理相关路由
"""
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from backend.config import UPLOAD_DIR, Config
from backend.services.data_service import DataService

data_bp = Blueprint('data', __name__, url_prefix='/api/data')
data_service = DataService()


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@data_bp.route('/upload', methods=['POST'])
def upload_dataset():
    """上传数据集"""
    try:
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = str(int(os.path.getmtime(__file__) * 1000))
        save_filename = f"dataset_{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_DIR, save_filename)
        file.save(file_path)
        
        # 解析文件
        result = data_service.upload_and_parse_file(file_path, filename)
        
        return jsonify({
            'success': True,
            'message': '数据集上传成功',
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_bp.route('/datasets', methods=['GET'])
def get_datasets():
    """获取所有数据集列表"""
    try:
        datasets = data_service.db.get_all_datasets()
        return jsonify({
            'success': True,
            'data': datasets
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset_info(dataset_id):
    """获取数据集详细信息"""
    try:
        info = data_service.get_dataset_info(dataset_id)
        return jsonify({
            'success': True,
            'data': info
        }), 200
    except ValueError as e:
        if "不存在" in str(e):
            return jsonify({'error': '数据集不存在'}), 404
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """删除数据集"""
    try:
        success = data_service.delete_dataset(dataset_id)
        if not success:
            return jsonify({'error': '数据集不存在'}), 404
            
        return jsonify({
            'success': True,
            'message': '数据集删除成功'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_bp.route('/datasets/batch', methods=['POST'])
def batch_delete_datasets():
    """批量删除数据集"""
    try:
        data = request.json
        dataset_ids = data.get('dataset_ids', [])
        
        if not dataset_ids:
            return jsonify({'error': '未提供dataset_ids'}), 400
            
        results = data_service.batch_delete_datasets(dataset_ids)
        
        return jsonify({
            'success': True,
            'message': '批量删除完成',
            'data': results
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@data_bp.route('/datasets/<int:dataset_id>/data', methods=['GET'])
def get_dataset_data(dataset_id):
    """获取数据集内容（分页）"""
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 100, type=int)
        
        data = data_service.get_dataset_data(dataset_id, page, page_size)
        
        return jsonify({
            'success': True,
            'data': data
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_bp.route('/datasets/<int:dataset_id>/data', methods=['POST'])
def save_dataset_data(dataset_id):
    """保存数据集内容"""
    try:
        data = request.json
        if not isinstance(data, list):
            return jsonify({'error': '数据格式错误，应为列表'}), 400
            
        data_service.save_dataset_data(dataset_id, data)
        
        return jsonify({
            'success': True,
            'message': '数据保存成功'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@data_bp.route('/columns/<int:column_id>', methods=['PUT'])
def update_column(column_id):
    """更新列属性"""
    try:
        properties = request.json
        data_service.update_column_properties(column_id, **properties)
        
        return jsonify({
            'success': True,
            'message': '列属性更新成功'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
