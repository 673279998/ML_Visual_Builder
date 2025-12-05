"""
Flask应用主入口
"""
import os
import sys
import webbrowser
from threading import Timer
from flask import Flask, send_from_directory
from flask_cors import CORS

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import Config
from backend.database.models import DatabaseModels

# 创建Flask应用
app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config.from_object(Config)

# 配置CORS
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}})

# 初始化数据库
DatabaseModels.create_tables()


@app.route('/')
def index():
    """主页路由"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health_check():
    """健康检查"""
    return {'status': 'ok', 'message': '服务运行正常'}


# 注册路由蓝图
from backend.routes.data_routes import data_bp
from backend.routes.model_routes import model_bp
from backend.routes.training_routes import training_bp
from backend.routes.prediction_routes import prediction_bp
from backend.routes.preprocessing_routes import preprocessing_bp
from backend.routes.workflow_routes import workflow_bp
from backend.routes.algorithm_routes import algorithm_bp
from backend.routes.hyperparameter_routes import hyperparameter_bp
from backend.routes.system_routes import system_bp

app.register_blueprint(data_bp)
app.register_blueprint(model_bp)
app.register_blueprint(training_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(preprocessing_bp)
app.register_blueprint(workflow_bp)
app.register_blueprint(algorithm_bp)
app.register_blueprint(hyperparameter_bp)
app.register_blueprint(system_bp)

def open_browser():
    """自动打开浏览器"""
    url = f'http://{Config.HOST}:{Config.PORT}'
    webbrowser.open(url)


if __name__ == '__main__':
    print("=" * 50)
    print("  机器学习可视化平台")
    print("=" * 50)
    print(f"  访问地址: http://{Config.HOST}:{Config.PORT}")
    print(f"  按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    # 延迟1.5秒后打开浏览器
    Timer(1.5, open_browser).start()
    
    # 启动Flask应用
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
