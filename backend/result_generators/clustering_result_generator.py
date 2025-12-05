"""
聚类算法结果生成器
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    silhouette_samples
)
from .base_result_generator import BaseResultGenerator


class ClusteringResultGenerator(BaseResultGenerator):
    """聚类算法结果生成器"""
    
    def __init__(self):
        super().__init__()
        self.result_type = "clustering"
    
    def generate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成聚类性能指标
        
        指标包括:
        - 轮廓系数 (silhouette_score)
        - Davies-Bouldin指数
        - Calinski-Harabasz指数
        - 簇内距离统计
        - 簇间距离统计
        """
        metrics = {}
        
        if X_test is None:
            return {'error': 'X_test is required for clustering metrics'}
        
        # 基础指标
        try:
            metrics['silhouette_score'] = float(silhouette_score(X_test, y_pred))
        except:
            metrics['silhouette_score'] = None
        
        try:
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X_test, y_pred))
        except:
            metrics['davies_bouldin_score'] = None
        
        try:
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X_test, y_pred))
        except:
            metrics['calinski_harabasz_score'] = None
        
        # 聚类统计
        unique_labels = np.unique(y_pred)
        metrics['n_clusters'] = int(len(unique_labels))
        metrics['n_noise_points'] = int(np.sum(y_pred == -1)) if -1 in y_pred else 0
        
        # 每个簇的样本数量
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(y_pred == label))
        metrics['cluster_sizes'] = cluster_sizes
        
        # 簇内距离统计
        intra_cluster_distances = {}
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
            cluster_points = X_test[y_pred == label]
            if len(cluster_points) > 1:
                # 计算到簇中心的平均距离
                center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                intra_cluster_distances[int(label)] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances))
                }
        metrics['intra_cluster_distances'] = intra_cluster_distances
        
        # 簇中心
        if hasattr(model, 'cluster_centers_'):
            metrics['cluster_centers'] = model.cluster_centers_.tolist()
        
        return metrics
    
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成聚类可视化数据
        
        可视化包括:
        - 轮廓图 (silhouette_plot)
        - 簇分布 (cluster_distribution)
        - 簇中心位置 (cluster_centers)
        - 二维投影散点图 (2d_projection)
        """
        visualizations = {}
        
        if X_test is None:
            return {'error': 'X_test is required for clustering visualizations'}
        
        # 1. 轮廓系数样本值
        try:
            silhouette_vals = silhouette_samples(X_test, y_pred)
            
            # 按簇组织轮廓值
            silhouette_by_cluster = {}
            for label in np.unique(y_pred):
                if label != -1:  # 跳过噪声点
                    cluster_silhouette = silhouette_vals[y_pred == label]
                    silhouette_by_cluster[int(label)] = {
                        'values': cluster_silhouette.tolist(),
                        'mean': float(np.mean(cluster_silhouette)),
                        'count': int(len(cluster_silhouette))
                    }
            
            visualizations['silhouette_plot'] = silhouette_by_cluster
        except Exception as e:
            visualizations['silhouette_error'] = str(e)
        
        # 2. 簇分布统计
        unique_labels = np.unique(y_pred)
        cluster_distribution = {}
        for label in unique_labels:
            cluster_distribution[int(label)] = int(np.sum(y_pred == label))
        
        visualizations['cluster_distribution'] = cluster_distribution
        
        # 3. 簇中心
        if hasattr(model, 'cluster_centers_'):
            visualizations['cluster_centers'] = {
                'centers': model.cluster_centers_.tolist(),
                'n_features': model.cluster_centers_.shape[1]
            }
        
        # 4. 二维投影(使用PCA降维到2D)
        if X_test.shape[1] > 2:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_test)
                
                # 限制点数
                max_points = 1000
                if len(X_2d) > max_points:
                    indices = np.random.choice(len(X_2d), max_points, replace=False)
                    X_2d = X_2d[indices]
                    labels = y_pred[indices]
                else:
                    labels = y_pred
                
                visualizations['2d_projection'] = {
                    'x': X_2d[:, 0].tolist(),
                    'y': X_2d[:, 1].tolist(),
                    'labels': labels.tolist(),
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                }
                
                # 投影后的簇中心
                if hasattr(model, 'cluster_centers_'):
                    centers_2d = pca.transform(model.cluster_centers_)
                    visualizations['2d_projection']['centers'] = {
                        'x': centers_2d[:, 0].tolist(),
                        'y': centers_2d[:, 1].tolist()
                    }
            except Exception as e:
                visualizations['2d_projection_error'] = str(e)
        elif X_test.shape[1] == 2:
            # 数据本身就是2D
            max_points = 1000
            if len(X_test) > max_points:
                indices = np.random.choice(len(X_test), max_points, replace=False)
                X_2d = X_test[indices]
                labels = y_pred[indices]
            else:
                X_2d = X_test
                labels = y_pred
            
            visualizations['2d_projection'] = {
                'x': X_2d[:, 0].tolist(),
                'y': X_2d[:, 1].tolist(),
                'labels': labels.tolist()
            }
            
            if hasattr(model, 'cluster_centers_'):
                visualizations['2d_projection']['centers'] = {
                    'x': model.cluster_centers_[:, 0].tolist(),
                    'y': model.cluster_centers_[:, 1].tolist()
                }
        
        # 5. 簇间距离矩阵
        if hasattr(model, 'cluster_centers_'):
            n_clusters = len(model.cluster_centers_)
            distance_matrix = np.zeros((n_clusters, n_clusters))
            
            for i in range(n_clusters):
                for j in range(n_clusters):
                    distance_matrix[i, j] = np.linalg.norm(
                        model.cluster_centers_[i] - model.cluster_centers_[j]
                    )
            
            visualizations['inter_cluster_distances'] = {
                'matrix': distance_matrix.tolist(),
                'cluster_labels': list(range(n_clusters))
            }
        
        # 6. 每个簇的特征统计
        cluster_feature_stats = {}
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = X_test[y_pred == label]
            cluster_feature_stats[int(label)] = {
                'mean': np.mean(cluster_points, axis=0).tolist(),
                'std': np.std(cluster_points, axis=0).tolist(),
                'min': np.min(cluster_points, axis=0).tolist(),
                'max': np.max(cluster_points, axis=0).tolist()
            }
        
        visualizations['cluster_feature_stats'] = cluster_feature_stats
        
        return visualizations
