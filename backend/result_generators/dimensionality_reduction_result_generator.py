"""
降维算法结果生成器
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error
from .base_result_generator import BaseResultGenerator


class DimensionalityReductionResultGenerator(BaseResultGenerator):
    """降维算法结果生成器"""
    
    def __init__(self):
        super().__init__()
        self.result_type = "dimensionality_reduction"
    
    def generate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成降维性能指标
        
        指标包括:
        - 方差解释率
        - 重构误差
        - 降维前后的维度
        """
        metrics = {}
        
        if X_test is None:
            return {'error': 'X_test is required for dimensionality reduction metrics'}
        
        # 降维后的数据就是y_pred
        X_reduced = y_pred
        
        # 基础统计
        metrics['original_dimensions'] = int(X_test.shape[1])
        metrics['reduced_dimensions'] = int(X_reduced.shape[1])
        metrics['dimension_reduction_ratio'] = float(X_reduced.shape[1] / X_test.shape[1])
        metrics['n_samples'] = int(X_test.shape[0])
        
        # 方差解释率(仅PCA等线性方法支持)
        if hasattr(model, 'explained_variance_ratio_'):
            metrics['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
            metrics['total_explained_variance'] = float(np.sum(model.explained_variance_ratio_))
            metrics['cumulative_explained_variance'] = np.cumsum(model.explained_variance_ratio_).tolist()
        
        # 重构误差(仅某些方法支持)
        if hasattr(model, 'inverse_transform'):
            try:
                X_reconstructed = model.inverse_transform(X_reduced)
                reconstruction_error = mean_squared_error(X_test, X_reconstructed)
                metrics['reconstruction_error'] = float(reconstruction_error)
                metrics['reconstruction_rmse'] = float(np.sqrt(reconstruction_error))
            except:
                metrics['reconstruction_error'] = None
        
        # 降维后数据的统计
        metrics['reduced_data_stats'] = {
            'mean': np.mean(X_reduced, axis=0).tolist(),
            'std': np.std(X_reduced, axis=0).tolist(),
            'min': np.min(X_reduced, axis=0).tolist(),
            'max': np.max(X_reduced, axis=0).tolist()
        }
        
        # 奇异值(PCA)
        if hasattr(model, 'singular_values_'):
            metrics['singular_values'] = model.singular_values_.tolist()
        
        # 组件(PCA)
        if hasattr(model, 'components_'):
            metrics['n_components'] = int(model.components_.shape[0])
        
        return metrics
    
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成降维可视化数据
        
        可视化包括:
        - 降维后的散点图
        - 方差解释率图
        - 成分载荷图
        - 重构误差分布
        """
        visualizations = {}
        
        if X_test is None:
            return {'error': 'X_test is required for dimensionality reduction visualizations'}
        
        X_reduced = y_pred
        
        # 1. 二维/三维散点图
        max_points = 1000
        if len(X_reduced) > max_points:
            indices = np.random.choice(len(X_reduced), max_points, replace=False)
            X_reduced_sample = X_reduced[indices]
        else:
            X_reduced_sample = X_reduced
        
        if X_reduced.shape[1] >= 2:
            visualizations['2d_scatter'] = {
                'x': X_reduced_sample[:, 0].tolist(),
                'y': X_reduced_sample[:, 1].tolist(),
                'component_labels': ['Component 1', 'Component 2']
            }
        
        # 生成3D散点图（如果数据维度不足3维，用0填充）
        x_coords = X_reduced_sample[:, 0].tolist()
        y_coords = (X_reduced_sample[:, 1] if X_reduced.shape[1] >= 2 else np.zeros(len(X_reduced_sample))).tolist()
        z_coords = (X_reduced_sample[:, 2] if X_reduced.shape[1] >= 3 else np.zeros(len(X_reduced_sample))).tolist()
        
        visualizations['3d_scatter'] = {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords,
            'component_labels': ['Component 1', 'Component 2', 'Component 3']
        }
        
        # 2. 方差解释率图(仅PCA等线性方法)
        if hasattr(model, 'explained_variance_ratio_'):
            visualizations['variance_explained'] = {
                'components': list(range(1, len(model.explained_variance_ratio_) + 1)),
                'variance_ratio': model.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(model.explained_variance_ratio_).tolist()
            }
        
        # 3. 成分载荷图(PCA)
        if hasattr(model, 'components_'):
            n_components = min(3, model.components_.shape[0])
            n_features = min(20, model.components_.shape[1])  # 限制特征数量
            
            visualizations['component_loadings'] = {
                'components': model.components_[:n_components, :n_features].tolist(),
                'n_components_shown': n_components,
                'n_features_shown': n_features,
                'feature_indices': list(range(n_features))
            }
        
        # 4. 重构误差分布
        if hasattr(model, 'inverse_transform'):
            try:
                X_reconstructed = model.inverse_transform(X_reduced)
                reconstruction_errors = np.sum((X_test - X_reconstructed) ** 2, axis=1)
                
                hist, bin_edges = np.histogram(reconstruction_errors, bins=30)
                visualizations['reconstruction_error_distribution'] = {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'mean_error': float(np.mean(reconstruction_errors)),
                    'median_error': float(np.median(reconstruction_errors))
                }
            except:
                pass
        
        # 5. 每个维度的分布
        dimension_distributions = []
        for dim in range(min(5, X_reduced.shape[1])):  # 最多5个维度
            hist, bin_edges = np.histogram(X_reduced[:, dim], bins=30)
            dimension_distributions.append({
                'dimension': dim + 1,
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mean': float(np.mean(X_reduced[:, dim])),
                'std': float(np.std(X_reduced[:, dim]))
            })
        
        visualizations['dimension_distributions'] = dimension_distributions
        
        # 6. 成分相关性矩阵
        if X_reduced.shape[1] <= 10:  # 只对较少维度计算
            correlation_matrix = np.corrcoef(X_reduced.T)
            visualizations['component_correlation'] = {
                'matrix': correlation_matrix.tolist(),
                'labels': [f'Component {i+1}' for i in range(X_reduced.shape[1])]
            }
        
        # 7. 降维效果对比
        # 原始数据的协方差矩阵特征值
        try:
            if X_test.shape[1] <= 100:  # 避免大矩阵计算
                cov_matrix = np.cov(X_test.T)
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                eigenvalues = sorted(eigenvalues, reverse=True)
                
                visualizations['eigenvalue_spectrum'] = {
                    'eigenvalues': eigenvalues[:20],  # 前20个特征值
                    'total_variance': float(np.sum(eigenvalues)),
                    'top_k_variance': float(np.sum(eigenvalues[:X_reduced.shape[1]]))
                }
        except:
            pass
        
        return visualizations
