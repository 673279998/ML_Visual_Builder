"""
回归算法结果生成器
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)
from .base_result_generator import BaseResultGenerator


class RegressionResultGenerator(BaseResultGenerator):
    """回归算法结果生成器"""
    
    def __init__(self):
        super().__init__()
        self.result_type = "regression"
    
    def generate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成回归性能指标
        
        指标包括:
        - MSE (均方误差)
        - RMSE (均方根误差)
        - MAE (平均绝对误差)
        - R² (决定系数)
        - 解释方差
        - MAPE (平均绝对百分比误差)
        """
        metrics = {}
        
        # 基础指标
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        
        # MAPE (避免除零)
        try:
            metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
        except:
            metrics['mape'] = None
        
        # 残差统计
        residuals = y_true - y_pred
        metrics['residual_mean'] = float(np.mean(residuals))
        metrics['residual_std'] = float(np.std(residuals))
        metrics['residual_min'] = float(np.min(residuals))
        metrics['residual_max'] = float(np.max(residuals))
        
        # 预测值统计
        metrics['prediction_mean'] = float(np.mean(y_pred))
        metrics['prediction_std'] = float(np.std(y_pred))
        metrics['prediction_min'] = float(np.min(y_pred))
        metrics['prediction_max'] = float(np.max(y_pred))
        
        # 真实值统计
        metrics['actual_mean'] = float(np.mean(y_true))
        metrics['actual_std'] = float(np.std(y_true))
        metrics['actual_min'] = float(np.min(y_true))
        metrics['actual_max'] = float(np.max(y_true))
        
        return metrics
    
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model: Any, X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        生成回归可视化数据
        
        可视化包括:
        - 预测vs实际散点图 (prediction_vs_actual)
        - 残差图 (residual_plot)
        - 残差分布直方图 (residual_distribution)
        - 预测误差分布 (error_distribution)
        """
        visualizations = {}
        
        # 限制可视化数据量
        max_points = 1000
        indices = np.random.choice(len(y_true), min(max_points, len(y_true)), replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        # 1. 预测vs实际散点图
        visualizations['prediction_vs_actual'] = {
            'actual': y_true_sample.tolist(),
            'predicted': y_pred_sample.tolist(),
            'perfect_line': {
                'x': [float(np.min(y_true)), float(np.max(y_true))],
                'y': [float(np.min(y_true)), float(np.max(y_true))]
            }
        }
        
        # 2. 残差图
        residuals = y_true - y_pred
        residuals_sample = residuals[indices]
        visualizations['residual_plot'] = {
            'predicted': y_pred_sample.tolist(),
            'residuals': residuals_sample.tolist(),
            'zero_line': {
                'x': [float(np.min(y_pred)), float(np.max(y_pred))],
                'y': [0, 0]
            }
        }
        
        # 3. 残差分布直方图
        hist, bin_edges = np.histogram(residuals, bins=30)
        visualizations['residual_distribution'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals))
        }
        
        # 4. 预测误差分布
        abs_errors = np.abs(residuals)
        error_hist, error_bins = np.histogram(abs_errors, bins=30)
        visualizations['error_distribution'] = {
            'counts': error_hist.tolist(),
            'bin_edges': error_bins.tolist(),
            'mean_error': float(np.mean(abs_errors)),
            'median_error': float(np.median(abs_errors))
        }
        
        # 5. 分位数-分位数图数据(Q-Q plot)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.random.normal(0, 1, len(sorted_residuals))
        theoretical_quantiles.sort()
        
        # 采样
        qq_indices = np.linspace(0, len(sorted_residuals)-1, min(100, len(sorted_residuals)), dtype=int)
        visualizations['qq_plot'] = {
            'theoretical_quantiles': theoretical_quantiles[qq_indices].tolist(),
            'sample_quantiles': sorted_residuals[qq_indices].tolist()
        }
        
        # 6. 误差百分比分布
        if np.all(y_true != 0):  # 避免除零
            percentage_errors = (residuals / y_true) * 100
            pct_hist, pct_bins = np.histogram(percentage_errors, bins=30)
            visualizations['percentage_error_distribution'] = {
                'counts': pct_hist.tolist(),
                'bin_edges': pct_bins.tolist(),
                'mean_pct_error': float(np.mean(percentage_errors)),
                'median_pct_error': float(np.median(percentage_errors))
            }
        
        # 7. 预测区间分析
        y_range = np.max(y_true) - np.min(y_true)
        n_bins = 10
        bins = np.linspace(np.min(y_true), np.max(y_true), n_bins + 1)
        
        bin_metrics = []
        for i in range(n_bins):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if np.sum(mask) > 0:
                bin_metrics.append({
                    'bin_start': float(bins[i]),
                    'bin_end': float(bins[i + 1]),
                    'count': int(np.sum(mask)),
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'r2': float(r2_score(y_true[mask], y_pred[mask])) if np.sum(mask) > 1 else None
                })
        
        visualizations['prediction_interval_analysis'] = bin_metrics
        
        return visualizations
