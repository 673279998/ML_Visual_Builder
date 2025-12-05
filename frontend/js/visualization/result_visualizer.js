/**
 * 结果可视化系统 - 使用Chart.js渲染各种图表
 */

class ResultVisualizer {
    constructor() {
        this.charts = new Map();  // 存储Chart实例
        this.init();
    }
    
    init() {
        // Chart.js全局配置
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = "'Segoe UI', 'Microsoft YaHei', sans-serif";
            Chart.defaults.color = '#666';
        }
    }
    
    /**
     * 渲染完整的训练结果
     */
    renderTrainingResults(containerId, results) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('容器不存在:', containerId);
            return;
        }
        
        container.innerHTML = '';
        
        // 根据算法类型渲染不同的可视化
        const algorithmType = results.algorithm_type;
        const metrics = results.complete_results?.metrics || results.performance_metrics || {};
        const visualizations = results.complete_results?.visualizations || {};
        
        // 渲染性能指标
        this.renderMetrics(container, algorithmType, metrics);
        
        // 渲染可视化图表
        if (algorithmType === 'classification') {
            this.renderClassificationVisualizations(container, visualizations);
        } else if (algorithmType === 'regression') {
            this.renderRegressionVisualizations(container, visualizations);
        } else if (algorithmType === 'clustering') {
            this.renderClusteringVisualizations(container, visualizations);
        } else if (algorithmType === 'dimensionality_reduction') {
            this.renderDimensionalityReductionVisualizations(container, visualizations);
        }
        
        // 渲染特征重要性
        if (results.feature_importance) {
            this.renderFeatureImportance(container, results.feature_importance);
        }
    }
    
    /**
     * 渲染性能指标卡片
     */
    renderMetrics(container, algorithmType, metrics) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>性能指标</h3>';
        
        const metricsGrid = document.createElement('div');
        metricsGrid.className = 'metrics-grid';
        
        // 根据算法类型显示不同的指标
        let metricsToShow = [];
        if (algorithmType === 'classification') {
            metricsToShow = [
                { key: 'accuracy', label: '准确率', format: 'percent' },
                { key: 'precision', label: '精确率', format: 'percent' },
                { key: 'recall', label: '召回率', format: 'percent' },
                { key: 'f1_score', label: 'F1分数', format: 'percent' },
                { key: 'roc_auc', label: 'ROC-AUC', format: 'number' }
            ];
        } else if (algorithmType === 'regression') {
            metricsToShow = [
                { key: 'mse', label: 'MSE', format: 'number' },
                { key: 'rmse', label: 'RMSE', format: 'number' },
                { key: 'mae', label: 'MAE', format: 'number' },
                { key: 'r2', label: 'R²分数', format: 'number' }
            ];
        } else if (algorithmType === 'clustering') {
            metricsToShow = [
                { key: 'silhouette_score', label: '轮廓系数', format: 'number' },
                { key: 'davies_bouldin_index', label: 'DB指数', format: 'number' },
                { key: 'calinski_harabasz_score', label: 'CH指数', format: 'number' }
            ];
        }
        
        metricsToShow.forEach(metric => {
            if (metrics[metric.key] !== undefined) {
                const card = document.createElement('div');
                card.className = 'metric-card';
                
                let value = metrics[metric.key];
                if (metric.format === 'percent') {
                    value = (value * 100).toFixed(2) + '%';
                } else {
                    value = typeof value === 'number' ? value.toFixed(4) : value;
                }
                
                card.innerHTML = `
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-value">${value}</div>
                `;
                metricsGrid.appendChild(card);
            }
        });
        
        section.appendChild(metricsGrid);
        container.appendChild(section);
    }
    
    /**
     * 渲染分类可视化
     */
    renderClassificationVisualizations(container, visualizations) {
        // 混淆矩阵
        if (visualizations.confusion_matrix) {
            this.renderConfusionMatrix(container, visualizations.confusion_matrix);
        }
        
        // ROC曲线
        if (visualizations.roc_curve) {
            this.renderROCCurve(container, visualizations.roc_curve);
        }
        
        // PR曲线
        if (visualizations.pr_curve) {
            this.renderPRCurve(container, visualizations.pr_curve);
        }
    }
    
    /**
     * 渲染混淆矩阵
     */
    renderConfusionMatrix(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>混淆矩阵</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'confusion-matrix-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const matrix = data.matrix;
        const labels = data.labels;
        
        // 使用热力图显示混淆矩阵
        const datasets = matrix.map((row, i) => ({
            label: `实际: ${labels[i]}`,
            data: row,
            backgroundColor: row.map((val, j) => {
                const max = Math.max(...matrix.flat());
                const alpha = val / max;
                return i === j ? `rgba(76, 175, 80, ${alpha})` : `rgba(244, 67, 54, ${alpha})`;
            })
        }));
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels.map(l => `预测: ${l}`),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: false },
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { stacked: false },
                    y: { stacked: false, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('confusion-matrix', chart);
    }
    
    /**
     * 渲染ROC曲线
     */
    renderROCCurve(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>ROC曲线 (AUC = ${data.auc.toFixed(4)})</h3>`;
        
        const canvas = document.createElement('canvas');
        canvas.id = 'roc-curve-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'ROC曲线',
                        data: data.fpr.map((x, i) => ({ x, y: data.tpr[i] })),
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: '随机分类器',
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        borderColor: '#999',
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { 
                        type: 'linear',
                        title: { display: true, text: '假阳性率 (FPR)' },
                        min: 0,
                        max: 1
                    },
                    y: { 
                        title: { display: true, text: '真阳性率 (TPR)' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
        
        this.charts.set('roc-curve', chart);
    }
    
    /**
     * 渲染PR曲线
     */
    renderPRCurve(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>精确率-召回率曲线</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'pr-curve-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'PR曲线',
                    data: data.recall.map((x, i) => ({ x, y: data.precision[i] })),
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { 
                        type: 'linear',
                        title: { display: true, text: '召回率 (Recall)' },
                        min: 0,
                        max: 1
                    },
                    y: { 
                        title: { display: true, text: '精确率 (Precision)' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
        
        this.charts.set('pr-curve', chart);
    }
    
    /**
     * 渲染回归可视化
     */
    renderRegressionVisualizations(container, visualizations) {
        // 预测vs实际
        if (visualizations.predicted_vs_actual) {
            this.renderPredictedVsActual(container, visualizations.predicted_vs_actual);
        }
        
        // 残差图
        if (visualizations.residuals_plot) {
            this.renderResidualsPlot(container, visualizations.residuals_plot);
        }
    }
    
    /**
     * 渲染预测vs实际散点图
     */
    renderPredictedVsActual(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>预测值 vs 实际值</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'predicted-actual-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: '预测结果',
                        data: data.actual.map((y, i) => ({ x: y, y: data.predicted[i] })),
                        backgroundColor: 'rgba(33, 150, 243, 0.5)'
                    },
                    {
                        label: '完美预测线',
                        data: [
                            { x: Math.min(...data.actual), y: Math.min(...data.actual) },
                            { x: Math.max(...data.actual), y: Math.max(...data.actual) }
                        ],
                        type: 'line',
                        borderColor: '#F44336',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { title: { display: true, text: '实际值' } },
                    y: { title: { display: true, text: '预测值' } }
                }
            }
        });
        
        this.charts.set('predicted-actual', chart);
    }
    
    /**
     * 渲染残差图
     */
    renderResidualsPlot(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>残差图</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'residuals-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '残差',
                    data: data.predicted.map((x, i) => ({ x, y: data.residuals[i] })),
                    backgroundColor: 'rgba(156, 39, 176, 0.5)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { title: { display: true, text: '预测值' } },
                    y: { 
                        title: { display: true, text: '残差' },
                        ticks: {
                            callback: (value) => value.toFixed(2)
                        }
                    }
                }
            }
        });
        
        this.charts.set('residuals', chart);
    }
    
    /**
     * 渲染聚类可视化
     */
    renderClusteringVisualizations(container, visualizations) {
        // 2D投影
        if (visualizations.scatter_2d) {
            this.renderClusterScatter(container, visualizations.scatter_2d);
        }
        
        // 轮廓图
        if (visualizations.silhouette_plot) {
            this.renderSilhouettePlot(container, visualizations.silhouette_plot);
        }
    }
    
    /**
     * 渲染聚类散点图
     */
    renderClusterScatter(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>聚类可视化 (PCA降维)</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'cluster-scatter-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        // 按簇分组数据
        const clusters = {};
        data.labels.forEach((label, i) => {
            if (!clusters[label]) {
                clusters[label] = [];
            }
            clusters[label].push({ x: data.x[i], y: data.y[i] });
        });
        
        const colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#FFEB3B'];
        const datasets = Object.entries(clusters).map(([label, points], i) => ({
            label: `簇 ${label}`,
            data: points,
            backgroundColor: colors[i % colors.length]
        }));
        
        const chart = new Chart(canvas, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { title: { display: true, text: 'PC1' } },
                    y: { title: { display: true, text: 'PC2' } }
                }
            }
        });
        
        this.charts.set('cluster-scatter', chart);
    }
    
    /**
     * 渲染轮廓图
     */
    renderSilhouettePlot(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>轮廓系数分析</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'silhouette-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: data.sample_indices,
                datasets: [{
                    label: '轮廓系数',
                    data: data.silhouette_values,
                    backgroundColor: data.silhouette_values.map(v => 
                        v >= 0 ? `rgba(76, 175, 80, ${v})` : `rgba(244, 67, 54, ${-v})`
                    )
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { display: false },
                    y: { 
                        title: { display: true, text: '轮廓系数' },
                        min: -1,
                        max: 1
                    }
                }
            }
        });
        
        this.charts.set('silhouette', chart);
    }
    
    /**
     * 渲染降维可视化
     */
    renderDimensionalityReductionVisualizations(container, visualizations) {
        // 2D散点图
        if (visualizations.scatter_2d) {
            this.renderDimReductionScatter(container, visualizations.scatter_2d);
        }
        
        // 方差解释率
        if (visualizations.explained_variance) {
            this.renderExplainedVariance(container, visualizations.explained_variance);
        }
    }
    
    /**
     * 渲染降维散点图
     */
    renderDimReductionScatter(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>降维结果可视化</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'dimred-scatter-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '样本点',
                    data: data.x.map((x, i) => ({ x, y: data.y[i] })),
                    backgroundColor: 'rgba(33, 150, 243, 0.5)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' }
                },
                scales: {
                    x: { title: { display: true, text: '维度 1' } },
                    y: { title: { display: true, text: '维度 2' } }
                }
            }
        });
        
        this.charts.set('dimred-scatter', chart);
    }
    
    /**
     * 渲染方差解释率
     */
    renderExplainedVariance(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>方差解释率</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'variance-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: data.ratios.map((_, i) => `PC${i + 1}`),
                datasets: [{
                    label: '方差解释率',
                    data: data.ratios.map(r => (r * 100).toFixed(2)),
                    backgroundColor: '#2196F3'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { title: { display: true, text: '主成分' } },
                    y: { 
                        title: { display: true, text: '方差解释率 (%)' },
                        beginAtZero: true
                    }
                }
            }
        });
        
        this.charts.set('variance', chart);
    }
    
    /**
     * 渲染特征重要性
     */
    renderFeatureImportance(container, importance) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>特征重要性</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'feature-importance-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        // 排序并取前10个
        const sorted = Object.entries(importance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: sorted.map(([name, _]) => name),
                datasets: [{
                    label: '重要性',
                    data: sorted.map(([_, value]) => value),
                    backgroundColor: '#FF9800'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { 
                        title: { display: true, text: '重要性分数' },
                        beginAtZero: true
                    }
                }
            }
        });
        
        this.charts.set('feature-importance', chart);
    }
    
    /**
     * 清空所有图表
     */
    clearAll() {
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
    }
}

// 导出
window.ResultVisualizer = ResultVisualizer;
