"""
数据库模型定义
"""
import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径，确保导入正确
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from backend.config import DATABASE_PATH
except ImportError:
    # 如果无法导入，直接计算路径
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    DATABASE_DIR = DATA_DIR / 'databases'
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH = DATABASE_DIR / 'ml_platform.db'


class DatabaseModels:
    """数据库模型类"""
    
    @staticmethod
    def create_tables():
        """创建所有数据表"""
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        # 1. datasets表(数据集表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_format TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                is_encoded BOOLEAN DEFAULT 0,
                source_dataset_id INTEGER,
                encoder_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (encoder_id) REFERENCES dataset_encoders(id)
            )
        ''')
        
        # 2. dataset_columns表(数据集列表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_columns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                column_name TEXT NOT NULL,
                column_index INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                is_target BOOLEAN DEFAULT 0,
                is_excluded BOOLEAN DEFAULT 0,
                encoding_method TEXT,
                encoding_config TEXT,
                missing_count INTEGER DEFAULT 0,
                unique_count INTEGER,
                statistics TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        ''')
        
        # 3. dataset_encoders表(数据集编码器表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_encoders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_dataset_id INTEGER NOT NULL,
                encoded_dataset_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                column_mappings TEXT NOT NULL,
                encoding_summary TEXT NOT NULL,
                workflow_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (encoded_dataset_id) REFERENCES datasets(id)
            )
        ''')

        # 检查并添加 workflow_id 列 (如果不存在)
        cursor.execute("PRAGMA table_info(dataset_encoders)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'workflow_id' not in columns:
            try:
                cursor.execute("ALTER TABLE dataset_encoders ADD COLUMN workflow_id TEXT")
                print("已添加 workflow_id 列到 dataset_encoders 表")
            except Exception as e:
                print(f"添加 workflow_id 列失败: {e}")
        
        # 4. workflows表(工作流表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                workflow_type TEXT NOT NULL,
                description TEXT,
                configuration TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 5. models表(模型表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                algorithm_type TEXT NOT NULL,
                algorithm_name TEXT NOT NULL,
                workflow_id INTEGER,
                dataset_id INTEGER NOT NULL,
                encoder_id INTEGER,
                model_file_path TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                dataset_schema TEXT NOT NULL,
                input_requirements TEXT NOT NULL,
                feature_importance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (encoder_id) REFERENCES dataset_encoders(id)
            )
        ''')
        
        # 6. preprocessing_components表(预处理组件表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessing_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,
                component_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                applied_columns TEXT NOT NULL,
                configuration TEXT NOT NULL,
                training_statistics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')
        
        # 7. training_results表(训练结果表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL UNIQUE,
                computed_results TEXT NOT NULL,
                visualization_data TEXT NOT NULL,
                training_log TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')
        
        # 8. predictions表(预测记录表)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                workflow_id INTEGER,
                input_dataset_id INTEGER NOT NULL,
                output_file_path TEXT NOT NULL,
                used_encoder BOOLEAN DEFAULT 0,
                used_preprocessors TEXT,
                prediction_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (workflow_id) REFERENCES workflows(id),
                FOREIGN KEY (input_dataset_id) REFERENCES datasets(id)
            )
        ''')
        
        # 创建索引以提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_columns_dataset_id ON dataset_columns(dataset_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_algorithm_type ON models(algorithm_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id)')
        
        conn.commit()
        conn.close()
        
        print("数据库表创建成功!")


if __name__ == '__main__':
    DatabaseModels.create_tables()
