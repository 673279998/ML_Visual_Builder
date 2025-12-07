"""
应用配置文件
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / 'data'
DATABASE_DIR = DATA_DIR / 'databases'
UPLOAD_DIR = DATA_DIR / 'uploads'
MODEL_DIR = DATA_DIR / 'models'
MODELS_DIR = MODEL_DIR  # 别名，保持一致性
ENCODER_DIR = DATA_DIR / 'encoders'
PREPROCESSOR_DIR = DATA_DIR / 'preprocessors'
VISUALIZATION_DIR = DATA_DIR / 'visualizations'
PREDICTIONS_DIR = DATA_DIR / 'predictions'  # 预测结果目录

# 确保所有目录存在
for dir_path in [DATABASE_DIR, UPLOAD_DIR, MODEL_DIR, ENCODER_DIR, PREPROCESSOR_DIR, VISUALIZATION_DIR, PREDICTIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据库配置
DATABASE_PATH = DATABASE_DIR / 'ml_platform.db'
DATABASE_URI = f'sqlite:///{DATABASE_PATH}'

# Flask配置
class Config:
    """Flask应用配置"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
    
    # 服务器配置
    HOST = 'localhost'
    PORT = 5002
    DEBUG = True
    
    # CORS配置
    CORS_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']
