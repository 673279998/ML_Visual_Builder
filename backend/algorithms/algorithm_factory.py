"""
算法工厂 - 用于创建和管理所有算法实例
"""
from typing import Dict, List, Any, Optional
from backend.algorithms.base_algorithm import BaseAlgorithm

# 导入分类算法
from backend.algorithms.classification.logistic_regression import LogisticRegression
from backend.algorithms.classification.random_forest_classifier import RandomForestClassifier
from backend.algorithms.classification.decision_tree_classifier import DecisionTreeClassifier
from backend.algorithms.classification.gradient_boosting_classifier import GradientBoostingClassifier
from backend.algorithms.classification.xgboost_classifier import XGBoostClassifier
from backend.algorithms.classification.lightgbm_classifier import LightGBMClassifier
from backend.algorithms.classification.catboost_classifier import CatBoostClassifier
from backend.algorithms.classification.svm_classifier import SVMClassifier
from backend.algorithms.classification.naive_bayes import NaiveBayesClassifier
from backend.algorithms.classification.knn_classifier import KNNClassifier
from backend.algorithms.classification.mlp_classifier import MLPClassifier

# 导入回归算法
from backend.algorithms.regression.linear_regression import LinearRegression
from backend.algorithms.regression.ridge_regression import RidgeRegression
from backend.algorithms.regression.lasso_regression import LassoRegression
from backend.algorithms.regression.elasticnet_regression import ElasticNetRegression
from backend.algorithms.regression.decision_tree_regressor import DecisionTreeRegressor
from backend.algorithms.regression.random_forest_regressor import RandomForestRegressor
from backend.algorithms.regression.gradient_boosting_regressor import GradientBoostingRegressor
from backend.algorithms.regression.xgboost_regressor import XGBoostRegressor
from backend.algorithms.regression.lightgbm_regressor import LightGBMRegressor
from backend.algorithms.regression.svr_regressor import SVRegressor
from backend.algorithms.regression.mlp_regressor import MLPRegressor

# 导入聚类算法
from backend.algorithms.clustering.kmeans import KMeansClustering
from backend.algorithms.clustering.hierarchical import HierarchicalClustering
from backend.algorithms.clustering.dbscan import DBSCANClustering
from backend.algorithms.clustering.gmm import GMMClustering
from backend.algorithms.clustering.spectral import SpectralClustering

# 导入降维算法
from backend.algorithms.dimensionality_reduction.pca import PCAReduction
from backend.algorithms.dimensionality_reduction.tsne import TSNEReduction
from backend.algorithms.dimensionality_reduction.umap_reduction import UMAPReduction
from backend.algorithms.dimensionality_reduction.lda import LDAReduction


class AlgorithmFactory:
    """算法工厂类"""
    
    # 算法注册表
    _algorithms = {
        # 分类算法(11种)
        'logistic_regression': LogisticRegression,
        'random_forest_classifier': RandomForestClassifier,
        'decision_tree_classifier': DecisionTreeClassifier,
        'gradient_boosting_classifier': GradientBoostingClassifier,
        'xgboost_classifier': XGBoostClassifier,
        'lightgbm_classifier': LightGBMClassifier,
        'catboost_classifier': CatBoostClassifier,
        'svm_classifier': SVMClassifier,
        'naive_bayes': NaiveBayesClassifier,
        'knn_classifier': KNNClassifier,
        'mlp_classifier': MLPClassifier,
        
        # 回归算法(11种)
        'linear_regression': LinearRegression,
        'ridge_regression': RidgeRegression,
        'lasso_regression': LassoRegression,
        'elasticnet_regression': ElasticNetRegression,
        'decision_tree_regressor': DecisionTreeRegressor,
        'random_forest_regressor': RandomForestRegressor,
        'gradient_boosting_regressor': GradientBoostingRegressor,
        'xgboost_regressor': XGBoostRegressor,
        'lightgbm_regressor': LightGBMRegressor,
        'svr': SVRegressor,
        'mlp_regressor': MLPRegressor,
        
        # 聚类算法(5种)
        'kmeans': KMeansClustering,
        'hierarchical': HierarchicalClustering,
        'dbscan': DBSCANClustering,
        'gmm': GMMClustering,
        'spectral': SpectralClustering,
        
        # 降维算法(4种)
        'pca': PCAReduction,
        'tsne': TSNEReduction,
        'umap': UMAPReduction,
        'lda': LDAReduction,
    }
    
    # 算法中文名称映射表
    _display_names = {
        # 分类算法
        'logistic_regression': '逻辑回归',
        'random_forest_classifier': '随机森林分类',
        'decision_tree_classifier': '决策树分类',
        'gradient_boosting_classifier': '梯度提升树分类',
        'xgboost_classifier': 'XGBoost分类',
        'lightgbm_classifier': 'LightGBM分类',
        'catboost_classifier': 'CatBoost分类',
        'svm_classifier': '支持向量机分类',
        'naive_bayes': '朴素贝叶斯',
        'knn_classifier': 'K近邻分类',
        'mlp_classifier': '多层感知机分类',
        
        # 回归算法
        'linear_regression': '线性回归',
        'ridge_regression': '岭回归',
        'lasso_regression': 'Lasso回归',
        'elasticnet_regression': 'ElasticNet回归',
        'decision_tree_regressor': '决策树回归',
        'random_forest_regressor': '随机森林回归',
        'gradient_boosting_regressor': '梯度提升树回归',
        'xgboost_regressor': 'XGBoost回归',
        'lightgbm_regressor': 'LightGBM回归',
        'svr': '支持向量回归',
        'mlp_regressor': '多层感知机回归',
        
        # 聚类算法
        'kmeans': 'K均值聚类',
        'hierarchical': '层次聚类',
        'dbscan': 'DBSCAN聚类',
        'gmm': '高斯混合模型',
        'spectral': '谱聚类',
        
        # 降维算法
        'pca': '主成分分析',
        'tsne': 't-SNE降维',
        'umap': 'UMAP降维',
        'lda': '线性判别分析',
    }
    
    # 旧名称映射（用于兼容）
    _legacy_mapping = {
        'decision_tree': 'decision_tree_classifier',
        'random_forest': 'random_forest_classifier',
        'svm': 'svm_classifier',
        'knn': 'knn_classifier',
        'gradient_boosting': 'gradient_boosting_classifier',
        'xgboost': 'xgboost_classifier',
        'lightgbm': 'lightgbm_classifier',
        'catboost': 'catboost_classifier',
        'mlp': 'mlp_classifier',
        'ridge': 'ridge_regression',
        'lasso': 'lasso_regression',
        'elastic_net': 'elasticnet_regression',
        'gaussian_mixture': 'gmm'
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str) -> BaseAlgorithm:
        """
        创建算法实例
        
        Args:
            algorithm_name: 算法名称
            
        Returns:
            算法实例
        """
        # 检查是否有旧名称映射
        if algorithm_name in cls._legacy_mapping:
            algorithm_name = cls._legacy_mapping[algorithm_name]

        if algorithm_name not in cls._algorithms:
            raise ValueError(f"未知的算法: {algorithm_name}")
        
        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class()
    
    @classmethod
    def get_algorithm_list(cls) -> List[Dict[str, Any]]:
        """
        获取所有可用算法列表
        
        Returns:
            算法列表
        """
        algorithms = []
        for name, algorithm_class in cls._algorithms.items():
            instance = algorithm_class()
            algorithms.append({
                'name': name,
                'type': instance.algorithm_type,
                'display_name': cls._display_names.get(name, name.replace('_', ' ').title())
            })
        return algorithms
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Dict[str, Any]:
        """
        获取算法详细信息
        
        Args:
            algorithm_name: 算法名称
            
        Returns:
            算法信息
        """
        algorithm = cls.create_algorithm(algorithm_name)
        return {
            'name': algorithm.algorithm_name,
            'type': algorithm.algorithm_type,
            'hyperparameters': algorithm.get_hyperparameters()
        }
    
    @classmethod
    def get_algorithms_by_type(cls, algorithm_type: str) -> List[Dict[str, Any]]:
        """
        根据类型获取算法列表
        
        Args:
            algorithm_type: 算法类型(classification/regression/clustering/reduction)
            
        Returns:
            算法列表
        """
        all_algorithms = cls.get_algorithm_list()
        return [alg for alg in all_algorithms if alg['type'] == algorithm_type]
    
    @classmethod
    def get_display_name(cls, algorithm_name: str) -> str:
        """
        获取算法的中文显示名称
        
        Args:
            algorithm_name: 算法内部名称
            
        Returns:
            中文显示名称
        """
        # 检查是否有旧名称映射
        if algorithm_name in cls._legacy_mapping:
            algorithm_name = cls._legacy_mapping[algorithm_name]
        
        return cls._display_names.get(algorithm_name, algorithm_name.replace('_', ' ').title())
