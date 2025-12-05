"""
超参数注册表 - 提供所有算法的超参数定义
"""

class HyperparameterRegistry:
    """算法超参数注册表"""
    
    def __init__(self):
        self.algorithms = self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """初始化算法列表和超参数定义"""
        return {
            'classification': [
                {
                    'name': 'logistic_regression',
                    'display_name': '逻辑回归',
                    'category': 'classification',
                    'hyperparameters': {
                        'C': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100},
                        'penalty': {'type': 'choice', 'default': 'l2', 'choices': ['l1', 'l2', 'elasticnet', 'none']},
                        'max_iter': {'type': 'int', 'default': 100, 'min': 50, 'max': 1000},
                        'solver': {'type': 'choice', 'default': 'lbfgs', 'choices': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}
                    }
                },
                {
                    'name': 'decision_tree_classifier',
                    'display_name': '决策树',
                    'category': 'classification',
                    'hyperparameters': {
                        'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50},
                        'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
                        'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 20}
                    }
                },
                {
                    'name': 'random_forest_classifier',
                    'display_name': '随机森林',
                    'category': 'classification',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50},
                        'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20}
                    }
                },
                {
                    'name': 'svm_classifier',
                    'display_name': '支持向量机',
                    'category': 'classification',
                    'hyperparameters': {
                        'C': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100},
                        'kernel': {'type': 'choice', 'default': 'rbf', 'choices': ['linear', 'poly', 'rbf', 'sigmoid']},
                        'gamma': {'type': 'choice', 'default': 'scale', 'choices': ['scale', 'auto']}
                    }
                },
                {
                    'name': 'knn_classifier',
                    'display_name': 'K近邻',
                    'category': 'classification',
                    'hyperparameters': {
                        'n_neighbors': {'type': 'int', 'default': 5, 'min': 1, 'max': 50},
                        'weights': {'type': 'choice', 'default': 'uniform', 'choices': ['uniform', 'distance']},
                        'metric': {'type': 'choice', 'default': 'minkowski', 'choices': ['euclidean', 'manhattan', 'minkowski']}
                    }
                },
                {
                    'name': 'naive_bayes',
                    'display_name': '朴素贝叶斯',
                    'category': 'classification',
                    'hyperparameters': {
                        'var_smoothing': {'type': 'float', 'default': 1e-9, 'min': 1e-10, 'max': 1e-5}
                    }
                },
                {
                    'name': 'gradient_boosting_classifier',
                    'display_name': '梯度提升树',
                    'category': 'classification',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
                        'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 20}
                    }
                },
                {
                    'name': 'xgboost_classifier',
                    'display_name': 'XGBoost',
                    'category': 'classification',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
                        'max_depth': {'type': 'int', 'default': 6, 'min': 1, 'max': 20}
                    }
                },
                {
                    'name': 'lightgbm_classifier',
                    'display_name': 'LightGBM',
                    'category': 'classification',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
                        'num_leaves': {'type': 'int', 'default': 31, 'min': 10, 'max': 100}
                    }
                },
                {
                    'name': 'catboost_classifier',
                    'display_name': 'CatBoost',
                    'category': 'classification',
                    'hyperparameters': {
                        'iterations': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
                        'depth': {'type': 'int', 'default': 6, 'min': 1, 'max': 16}
                    }
                },
                {
                    'name': 'mlp_classifier',
                    'display_name': '多层感知机',
                    'category': 'classification',
                    'hyperparameters': {
                        'hidden_layer_sizes': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000},
                        'learning_rate_init': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1},
                        'max_iter': {'type': 'int', 'default': 200, 'min': 50, 'max': 1000}
                    }
                }
            ],
            'regression': [
                {
                    'name': 'linear_regression',
                    'display_name': '线性回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'fit_intercept': {'type': 'bool', 'default': True}
                    }
                },
                {
                    'name': 'ridge_regression',
                    'display_name': '岭回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'alpha': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100}
                    }
                },
                {
                    'name': 'lasso_regression',
                    'display_name': 'Lasso回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'alpha': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100}
                    }
                },
                {
                    'name': 'elasticnet_regression',
                    'display_name': '弹性网络',
                    'category': 'regression',
                    'hyperparameters': {
                        'alpha': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100},
                        'l1_ratio': {'type': 'float', 'default': 0.5, 'min': 0, 'max': 1}
                    }
                },
                {
                    'name': 'svr',
                    'display_name': '支持向量回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'C': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100},
                        'kernel': {'type': 'choice', 'default': 'rbf', 'choices': ['linear', 'poly', 'rbf', 'sigmoid']},
                        'epsilon': {'type': 'float', 'default': 0.1, 'min': 0.001, 'max': 1.0}
                    }
                },
                {
                    'name': 'decision_tree_regressor',
                    'display_name': '决策树回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50},
                        'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20}
                    }
                },
                {
                    'name': 'random_forest_regressor',
                    'display_name': '随机森林回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50}
                    }
                },
                {
                    'name': 'gradient_boosting_regressor',
                    'display_name': '梯度提升回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0}
                    }
                },
                {
                    'name': 'xgboost_regressor',
                    'display_name': 'XGBoost回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0}
                    }
                },
                {
                    'name': 'lightgbm_regressor',
                    'display_name': 'LightGBM回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
                        'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
                        'num_leaves': {'type': 'int', 'default': 31, 'min': 10, 'max': 100}
                    }
                },
                {
                    'name': 'mlp_regressor',
                    'display_name': '多层感知机回归',
                    'category': 'regression',
                    'hyperparameters': {
                        'hidden_layer_sizes': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000},
                        'learning_rate_init': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1},
                        'max_iter': {'type': 'int', 'default': 200, 'min': 50, 'max': 1000}
                    }
                }
            ],
            'clustering': [
                {
                    'name': 'kmeans',
                    'display_name': 'K均值聚类',
                    'category': 'clustering',
                    'hyperparameters': {
                        'n_clusters': {'type': 'int', 'default': 3, 'min': 2, 'max': 20},
                        'max_iter': {'type': 'int', 'default': 300, 'min': 50, 'max': 1000}
                    }
                },
                {
                    'name': 'dbscan',
                    'display_name': 'DBSCAN',
                    'category': 'clustering',
                    'hyperparameters': {
                        'eps': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 5.0},
                        'min_samples': {'type': 'int', 'default': 5, 'min': 1, 'max': 50}
                    }
                },
                {
                    'name': 'hierarchical',
                    'display_name': '层次聚类',
                    'category': 'clustering',
                    'hyperparameters': {
                        'n_clusters': {'type': 'int', 'default': 3, 'min': 2, 'max': 20},
                        'linkage': {'type': 'choice', 'default': 'ward', 'choices': ['ward', 'complete', 'average', 'single']}
                    }
                },
                {
                    'name': 'gmm',
                    'display_name': '高斯混合模型',
                    'category': 'clustering',
                    'hyperparameters': {
                        'n_components': {'type': 'int', 'default': 3, 'min': 2, 'max': 20},
                        'covariance_type': {'type': 'choice', 'default': 'full', 'choices': ['full', 'tied', 'diag', 'spherical']}
                    }
                },
                {
                    'name': 'spectral',
                    'display_name': '谱聚类',
                    'category': 'clustering',
                    'hyperparameters': {
                        'n_clusters': {'type': 'int', 'default': 3, 'min': 2, 'max': 20},
                        'affinity': {'type': 'choice', 'default': 'rbf', 'choices': ['rbf', 'nearest_neighbors', 'precomputed']}
                    }
                }
            ],
            'dimensionality_reduction': [
                {
                    'name': 'pca',
                    'display_name': '主成分分析',
                    'category': 'dimensionality_reduction',
                    'hyperparameters': {
                        'n_components': {'type': 'int', 'default': 2, 'min': 1, 'max': 50}
                    }
                },
                {
                    'name': 'tsne',
                    'display_name': 't-SNE',
                    'category': 'dimensionality_reduction',
                    'hyperparameters': {
                        'n_components': {'type': 'int', 'default': 2, 'min': 1, 'max': 3},
                        'perplexity': {'type': 'float', 'default': 30.0, 'min': 5.0, 'max': 50.0}
                    }
                },
                {
                    'name': 'umap',
                    'display_name': 'UMAP',
                    'category': 'dimensionality_reduction',
                    'hyperparameters': {
                        'n_components': {'type': 'int', 'default': 2, 'min': 1, 'max': 50},
                        'n_neighbors': {'type': 'int', 'default': 15, 'min': 2, 'max': 100}
                    }
                },
                {
                    'name': 'lda',
                    'display_name': '线性判别分析',
                    'category': 'dimensionality_reduction',
                    'hyperparameters': {
                        'n_components': {'type': 'int', 'default': 2, 'min': 1, 'max': 20}
                    }
                }
            ]
        }
    
    def get_all_algorithms(self):
        """获取所有算法"""
        return self.algorithms
    
    def get_algorithms_by_type(self, algorithm_type):
        """按类型获取算法"""
        return self.algorithms.get(algorithm_type, [])
    
    def get_hyperparameters(self, algorithm_name):
        """获取指定算法的超参数"""
        for category in self.algorithms.values():
            for alg in category:
                if alg['name'] == algorithm_name:
                    return alg.get('hyperparameters', {})
        return {}
    
    def get_algorithm_info(self, algorithm_name):
        """获取算法完整信息"""
        for category in self.algorithms.values():
            for alg in category:
                if alg['name'] == algorithm_name:
                    return alg
        return None
    
    @staticmethod
    def get_param_grid(algorithm_name):
        """
        获取算法的默认参数网格（用于网格搜索）
        """
        registry = HyperparameterRegistry()
        hyperparams = registry.get_hyperparameters(algorithm_name)
        param_grid = {}
        
        for name, config in hyperparams.items():
            if config['type'] == 'int':
                # 对于整数，生成一个范围或列表
                if 'min' in config and 'max' in config:
                    start = config['min']
                    end = config['max']
                    # 简单的启发式生成: 如果范围小，取所有值；如果范围大，取几个点
                    if end - start < 5:
                         param_grid[name] = list(range(start, end + 1))
                    else:
                         # 生成5个等间距的点，并取整
                         step = max(1, (end - start) // 4)
                         param_grid[name] = list(range(start, end + 1, step))
                elif 'choices' in config:
                     param_grid[name] = config['choices']
                     
            elif config['type'] == 'float':
                # 对于浮点数
                if 'min' in config and 'max' in config:
                    # 生成5个等间距的点
                    import numpy as np
                    param_grid[name] = np.linspace(config['min'], config['max'], 5).tolist()
                    
            elif config['type'] == 'choice':
                if 'choices' in config:
                    param_grid[name] = config['choices']
                    
            elif config['type'] == 'bool':
                param_grid[name] = [True, False]
                
        return param_grid

    @staticmethod
    def get_param_distributions(algorithm_name):
        """
        获取算法的参数分布（用于随机搜索）
        """
        registry = HyperparameterRegistry()
        hyperparams = registry.get_hyperparameters(algorithm_name)
        param_dist = {}
        
        for name, config in hyperparams.items():
            if config['type'] == 'int':
                if 'min' in config and 'max' in config:
                    from scipy.stats import randint
                    param_dist[name] = randint(config['min'], config['max'] + 1)
                    
            elif config['type'] == 'float':
                if 'min' in config and 'max' in config:
                    from scipy.stats import uniform
                    param_dist[name] = uniform(config['min'], config['max'] - config['min'])
                    
            elif config['type'] == 'choice':
                if 'choices' in config:
                    param_dist[name] = config['choices']
                    
            elif config['type'] == 'bool':
                param_dist[name] = [True, False]
                
        return param_dist
