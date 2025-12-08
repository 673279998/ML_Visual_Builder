"""
超参数调优服务
Hyperparameter Tuning Service
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, r2_score
from sklearn.utils.multiclass import type_of_target
import logging
import time

from backend.algorithms.algorithm_factory import AlgorithmFactory
from backend.utils.json_utils import sanitize_for_json

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self):
        self.best_params = {}
        self.best_score = None
        self.cv_results = {}
        self.search_history = []
    
    def _prepare_target(self, y: np.ndarray, algorithm_type: str) -> np.ndarray:
        """
        准备目标变量
        针对分类任务，确保目标变量为数值类型
        """
        if algorithm_type == 'classification':
            try:
                # 确保y是1D
                if len(y.shape) > 1:
                    y = y.ravel()
                    
                target_type = type_of_target(y)
                logger.info(f"目标变量类型检测: {target_type}")
                
                # 对于分类任务，确保目标变量为数值类型
                # 如果目标变量是字符串或非数值类型，进行LabelEncoding转换
                if target_type in ['binary', 'multiclass']:
                    # 检查是否需要转换
                    if y.dtype.kind in 'OSU':  # 字符串类型
                        logger.info("检测到分类任务的目标变量为字符串类型，进行LabelEncoding转换")
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        logger.info(f"LabelEncoding转换完成，类别: {le.classes_}")
                    elif target_type == 'continuous':
                        # 连续值转换为离散值（如0.0, 1.0 -> 0, 1）
                        logger.info("检测到分类任务的目标变量为连续值，尝试转换为离散值")
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        logger.info(f"LabelEncoding转换完成，类别: {le.classes_}")
            except Exception as e:
                logger.warning(f"目标变量转换失败: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        return y

    def grid_search(
        self,
        algorithm_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List],
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        网格搜索
        
        Args:
            algorithm_name: 算法名称
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数
            scoring: 评分方法
            n_jobs: 并行任务数(-1表示使用所有CPU)
            
        Returns:
            搜索结果
        """
        logger.info(f"开始网格搜索: {algorithm_name}")
        logger.info(f"DEBUG: grid_search received param_grid: {param_grid}")
        start_time = time.time()
        
        # 创建算法实例
        algorithm = AlgorithmFactory.create_algorithm(algorithm_name)
        
        # 预处理目标变量
        y_train = self._prepare_target(y_train, algorithm.algorithm_type)

        # 自动选择评分方法
        target_type = type_of_target(y_train)
        
        if scoring is None:
            # 优先根据目标变量类型选择
            if target_type == 'continuous':
                scoring = 'r2'
            elif target_type in ['binary', 'multiclass']:
                scoring = 'accuracy'
            # 如果无法确定，则根据算法类型选择
            elif algorithm.algorithm_type == 'classification':
                scoring = 'accuracy'
            elif algorithm.algorithm_type == 'regression':
                scoring = 'r2'
            else:
                scoring = 'accuracy'
        elif target_type == 'continuous' and scoring == 'accuracy':
            logger.warning("目标变量为连续值(回归任务)，不支持accuracy评分，已自动修正为r2")
            scoring = 'r2'
        
        # 创建GridSearchCV
        # 获取estimator
        estimator = None
        if hasattr(algorithm, 'get_estimator') and algorithm.get_estimator() is not None:
            estimator = algorithm.get_estimator()
        elif hasattr(algorithm, 'model') and algorithm.model is not None:
            estimator = algorithm.model
        else:
            # 尝试直接使用算法实例(如果实现了sklearn接口)
            estimator = algorithm
            
        if estimator is None:
            raise ValueError(f"算法 {algorithm_name} 未实现 get_estimator() 方法，且未初始化模型，无法进行超参数调优")
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # 执行搜索
        grid_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.cv_results = grid_search.cv_results_
        
        elapsed_time = time.time() - start_time
        
        result = {
            'method': 'grid_search',
            'algorithm_name': algorithm_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_combinations': len(grid_search.cv_results_['params']),
            'cv_folds': cv,
            'scoring': scoring,
            'elapsed_time': elapsed_time,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
        
        self.search_history.append(result)
        
        logger.info(f"网格搜索完成,最佳参数: {self.best_params}")
        logger.info(f"最佳得分: {self.best_score:.4f}")
        logger.info(f"耗时: {elapsed_time:.2f}秒")
        
        return sanitize_for_json(result)
    
    def random_search(
        self,
        algorithm_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        随机搜索
        
        Args:
            algorithm_name: 算法名称
            X_train: 训练特征
            y_train: 训练标签
            param_distributions: 参数分布
            n_iter: 迭代次数
            cv: 交叉验证折数
            scoring: 评分方法
            n_jobs: 并行任务数
            random_state: 随机种子
            
        Returns:
            搜索结果
        """
        logger.info(f"开始随机搜索: {algorithm_name}")
        start_time = time.time()
        
        # 创建算法实例
        algorithm = AlgorithmFactory.create_algorithm(algorithm_name)
        
        # 预处理目标变量
        y_train = self._prepare_target(y_train, algorithm.algorithm_type)
        
        # 自动选择评分方法
        target_type = type_of_target(y_train)
        
        if scoring is None:
            # 优先根据目标变量类型选择
            if target_type == 'continuous':
                scoring = 'r2'
            elif target_type in ['binary', 'multiclass']:
                scoring = 'accuracy'
            elif algorithm.algorithm_type == 'classification':
                scoring = 'accuracy'
            elif algorithm.algorithm_type == 'regression':
                scoring = 'r2'
        elif target_type == 'continuous' and scoring == 'accuracy':
            logger.warning("目标变量为连续值(回归任务)，不支持accuracy评分，已自动修正为r2")
            scoring = 'r2'
        
        # 创建RandomizedSearchCV
        # 获取estimator
        estimator = None
        if hasattr(algorithm, 'get_estimator') and algorithm.get_estimator() is not None:
            estimator = algorithm.get_estimator()
        elif hasattr(algorithm, 'model') and algorithm.model is not None:
            estimator = algorithm.model
        else:
            estimator = algorithm

        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=random_state,
            return_train_score=True
        )
        
        # 执行搜索
        random_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = random_search.cv_results_
        
        elapsed_time = time.time() - start_time
        
        result = {
            'method': 'random_search',
            'algorithm_name': algorithm_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iter': n_iter,
            'cv_folds': cv,
            'scoring': scoring,
            'elapsed_time': elapsed_time,
            'cv_results': {
                'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': random_search.cv_results_['std_test_score'].tolist(),
                'params': random_search.cv_results_['params']
            }
        }
        
        self.search_history.append(result)
        
        logger.info(f"随机搜索完成,最佳参数: {self.best_params}")
        logger.info(f"最佳得分: {self.best_score:.4f}")
        logger.info(f"耗时: {elapsed_time:.2f}秒")
        
        return sanitize_for_json(result)
    
    def bayesian_optimization(
        self,
        algorithm_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Dict[str, Tuple],
        n_iter: int = 50,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        贝叶斯优化
        
        Args:
            algorithm_name: 算法名称
            X_train: 训练特征
            y_train: 训练标签
            param_space: 参数空间
            n_iter: 迭代次数
            cv: 交叉验证折数
            scoring: 评分方法
            n_jobs: 并行任务数
            random_state: 随机种子
            
        Returns:
            搜索结果
        """
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer, Categorical
            from scipy.stats import uniform, randint, loguniform
            
            # 将 scipy.stats 分布转换为 skopt 空间
            skopt_space = {}
            for key, value in param_space.items():
                if hasattr(value, 'dist'):
                    # 处理 scipy 分布
                    if value.dist.name == 'uniform':
                        # uniform(loc, scale) -> Real(loc, loc + scale)
                        loc = value.args[0] if len(value.args) > 0 else value.kwds.get('loc', 0)
                        scale = value.args[1] if len(value.args) > 1 else value.kwds.get('scale', 1)
                        skopt_space[key] = Real(loc, loc + scale, prior='uniform')
                    elif value.dist.name == 'randint':
                        # randint(low, high) -> Integer(low, high)
                        low = value.args[0] if len(value.args) > 0 else value.kwds.get('low', 0)
                        high = value.args[1] if len(value.args) > 1 else value.kwds.get('high', 10)
                        skopt_space[key] = Integer(low, high)
                    elif value.dist.name == 'loguniform':
                        # loguniform(a, b) -> Real(a, b, prior='log-uniform')
                        a = value.args[0] if len(value.args) > 0 else value.kwds.get('a', 1e-6)
                        b = value.args[1] if len(value.args) > 1 else value.kwds.get('b', 1e-2)
                        skopt_space[key] = Real(a, b, prior='log-uniform')
                    else:
                        # 默认回退到列表，虽然可能不准确
                        logger.warning(f"Unsupported distribution {value.dist.name} for {key}, using original")
                        skopt_space[key] = value
                elif isinstance(value, list):
                    # 列表 -> Categorical
                    skopt_space[key] = Categorical(value)
                else:
                    # 默认情况，如果不是list也不是dist，可能是单值
                    # skopt BayesSearchCV 也可以接受 Categorical([value]) 来表示固定值
                    skopt_space[key] = Categorical([value]) if not isinstance(value, list) else Categorical(value)
                    
            param_space = skopt_space
            print(f"DEBUG: Converted skopt_space: {skopt_space}")
            
        except ImportError:
            logger.warning("scikit-optimize未安装,使用随机搜索替代")
            # 将贝叶斯参数空间转换为随机搜索格式
            param_distributions = {}
            for key, value in param_space.items():
                # 处理单值列表（即固定参数）
                if isinstance(value, list) and len(value) == 1:
                    param_distributions[key] = value
                # 处理多值列表
                elif isinstance(value, list) and len(value) > 1:
                    param_distributions[key] = value
                # 处理元组（通常是范围）
                elif isinstance(value, tuple) and len(value) == 2:
                    param_distributions[key] = value
            return self.random_search(
                algorithm_name, X_train, y_train,
                param_distributions, n_iter, cv, scoring, random_state=random_state
            )
        
        logger.info(f"开始贝叶斯优化: {algorithm_name}")
        logger.info(f"DEBUG: bayesian_optimization received param_space: {param_space}")
        start_time = time.time()
        
        # 创建算法实例
        algorithm = AlgorithmFactory.create_algorithm(algorithm_name)
        
        # 预处理目标变量
        y_train = self._prepare_target(y_train, algorithm.algorithm_type)
        
        # 自动选择评分方法
        target_type = type_of_target(y_train)
        
        if scoring is None:
            # 优先根据目标变量类型选择
            if target_type == 'continuous':
                scoring = 'r2'
            elif target_type in ['binary', 'multiclass']:
                scoring = 'accuracy'
            elif algorithm.algorithm_type == 'classification':
                scoring = 'accuracy'
            elif algorithm.algorithm_type == 'regression':
                scoring = 'r2'
        elif target_type == 'continuous' and scoring == 'accuracy':
            logger.warning("目标变量为连续值(回归任务)，不支持accuracy评分，已自动修正为r2")
            scoring = 'r2'
        
        # 创建BayesSearchCV
        # 获取estimator
        estimator = None
        if hasattr(algorithm, 'get_estimator') and algorithm.get_estimator() is not None:
            estimator = algorithm.get_estimator()
        elif hasattr(algorithm, 'model') and algorithm.model is not None:
            estimator = algorithm.model
        else:
            estimator = algorithm

        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=random_state,
            return_train_score=True
        )
        
        # 执行搜索
        bayes_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params = bayes_search.best_params_
        self.best_score = bayes_search.best_score_
        self.cv_results = bayes_search.cv_results_
        
        elapsed_time = time.time() - start_time
        
        result = {
            'method': 'bayesian_optimization',
            'algorithm_name': algorithm_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iter': n_iter,
            'cv_folds': cv,
            'scoring': scoring,
            'elapsed_time': elapsed_time,
            'cv_results': {
                'mean_test_score': bayes_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': bayes_search.cv_results_['std_test_score'].tolist(),
                'params': bayes_search.cv_results_['params']
            }
        }
        
        self.search_history.append(result)
        
        logger.info(f"贝叶斯优化完成,最佳参数: {self.best_params}")
        logger.info(f"最佳得分: {self.best_score:.4f}")
        logger.info(f"耗时: {elapsed_time:.2f}秒")
        
        return sanitize_for_json(result)
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """获取调优总结"""
        return {
            'total_searches': len(self.search_history),
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_history': self.search_history
        }
    
    def get_param_importance(self) -> Dict[str, float]:
        """
        获取参数重要性
        
        Returns:
            参数重要性字典
        """
        if not self.cv_results:
            return {}
        
        # 简单分析:计算每个参数值变化对得分的影响
        param_importance = {}
        
        # TODO: 实现更复杂的参数重要性分析
        
        return param_importance


class HyperparameterRegistry:
    """超参数注册表 - 为每种算法提供推荐的参数空间"""
    
    @staticmethod
    def get_param_grid(algorithm_name: str) -> Dict[str, List]:
        """
        获取算法的参数网格(用于网格搜索)
        
        Args:
            algorithm_name: 算法名称
            
        Returns:
            参数网格
        """
        param_grids = {
            # ==========================================
            # 分类算法 (Classification Algorithms)
            # ==========================================
            'logistic_regression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest_classifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'decision_tree_classifier': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting_classifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost_classifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            },
            'lightgbm_classifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'max_depth': [-1, 10, 20],
                'min_child_samples': [20, 30, 50]
            },
            'catboost_classifier': {
                'iterations': [100, 200, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7]
            },
            'svm_classifier': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'degree': [3, 4, 5]
            },
            'naive_bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'knn_classifier': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'mlp_classifier': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },

            # ==========================================
            # 回归算法 (Regression Algorithms)
            # ==========================================
            'linear_regression': {
                'fit_intercept': [True, False]
            },
            'ridge_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso_regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'fit_intercept': [True, False]
            },
            'elasticnet_regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'fit_intercept': [True, False]
            },
            'random_forest_regressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'decision_tree_regressor': {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting_regressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost_regressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lightgbm_regressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'max_depth': [-1, 10, 20],
                'min_child_samples': [20, 30, 50]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.01, 0.1, 0.5]
            },
            'mlp_regressor': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        return param_grids.get(algorithm_name, {})
    
    @staticmethod
    def get_param_distributions(algorithm_name: str) -> Dict[str, Any]:
        """
        获取算法的参数分布(用于随机搜索)
        
        Args:
            algorithm_name: 算法名称
            
        Returns:
            参数分布
        """
        from scipy.stats import randint, uniform
        
        param_distributions = {
            # ==========================================
            # 分类算法 (Classification Algorithms)
            # ==========================================
            'logistic_regression': {
                'C': uniform(0.01, 10.0),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest_classifier': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            },
            'decision_tree_classifier': {
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'criterion': ['gini', 'entropy']
            },
            'gradient_boosting_classifier': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            },
            'xgboost_classifier': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            },
            'lightgbm_classifier': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 150),
                'max_depth': randint(-1, 30),
                'min_child_samples': randint(10, 100)
            },
            'catboost_classifier': {
                'iterations': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'depth': randint(4, 12),
                'l2_leaf_reg': uniform(1, 10)
            },
            'svm_classifier': {
                'C': uniform(0.1, 100.0),
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'naive_bayes': {
                'var_smoothing': uniform(1e-10, 1e-5)
            },
            'knn_classifier': {
                'n_neighbors': randint(2, 20),
                'leaf_size': randint(10, 50),
                'weights': ['uniform', 'distance']
            },
            'mlp_classifier': {
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd'],
                'alpha': uniform(0.0001, 0.1),
                'learning_rate': ['constant', 'adaptive']
            },

            # ==========================================
            # 回归算法 (Regression Algorithms)
            # ==========================================
            'linear_regression': {
                'fit_intercept': [True, False]
            },
            'ridge_regression': {
                'alpha': uniform(0.1, 100.0)
            },
            'lasso_regression': {
                'alpha': uniform(0.001, 10.0)
            },
            'elasticnet_regression': {
                'alpha': uniform(0.001, 10.0),
                'l1_ratio': uniform(0.1, 0.9)
            },
            'random_forest_regressor': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            },
            'decision_tree_regressor': {
                'max_depth': randint(5, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
            },
            'gradient_boosting_regressor': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            },
            'xgboost_regressor': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            },
            'lightgbm_regressor': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 150),
                'max_depth': randint(-1, 30),
                'min_child_samples': randint(10, 100)
            },
            'svr': {
                'C': uniform(0.1, 100.0),
                'epsilon': uniform(0.01, 0.5),
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf', 'poly']
            },
            'mlp_regressor': {
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd'],
                'alpha': uniform(0.0001, 0.1),
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        return param_distributions.get(algorithm_name, {})
