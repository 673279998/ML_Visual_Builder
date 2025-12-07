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
        
        // 渲染算法特定信息
        if (results.complete_results?.algorithm_specific_info) {
            this.renderAlgorithmSpecificInfo(container, results.complete_results.algorithm_specific_info, results.algorithm_name);
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
        } else if (visualizations.roc_curve_multiclass) {
            // 多分类ROC曲线
            this.renderMulticlassROCCurve(container, visualizations.roc_curve_multiclass);
        }
        
        // PR曲线
        if (visualizations.pr_curve) {
            this.renderPRCurve(container, visualizations.pr_curve);
        }
        
        // 错误分析
        if (visualizations.error_analysis) {
            this.renderErrorAnalysis(container, visualizations.error_analysis);
        }
        
        // 预测概率分布
        if (visualizations.probability_distribution) {
            this.renderProbabilityDistribution(container, visualizations.probability_distribution);
        }
        
        // 类别预测分布
        if (visualizations.class_prediction_distribution) {
            this.renderClassPredictionDistribution(container, visualizations.class_prediction_distribution);
        }
    }
    
    /**
     * 渲染混淆矩阵
     */
    renderConfusionMatrix(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>混淆矩阵</h3>';
        
        const matrix = data.matrix;
        const labels = data.labels;
        
        // 创建HTML表格显示混淆矩阵
        const tableContainer = document.createElement('div');
        tableContainer.style.cssText = 'overflow-x: auto; margin-top: 10px;';
        
        let tableHTML = '<table class="confusion-matrix-table" style="border-collapse: collapse; margin: 0 auto; background: white;">';
        
        // 表头
        tableHTML += '<thead><tr><th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;"></th>';
        labels.forEach(label => {
            tableHTML += `<th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5; font-weight: bold;">预测: ${label}</th>`;
        });
        tableHTML += '</tr></thead>';
        
        // 表体 - 热力图效果
        tableHTML += '<tbody>';
        const maxVal = Math.max(...matrix.flat());
        
        matrix.forEach((row, i) => {
            tableHTML += '<tr>';
            tableHTML += `<th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5; font-weight: bold;">实际: ${labels[i]}</th>`;
            
            row.forEach((val, j) => {
                // 计算颜色深度
                const intensity = maxVal > 0 ? val / maxVal : 0;
                // 对角线（正确预测）用绿色，其他用红色
                const color = i === j 
                    ? `rgba(76, 175, 80, ${0.2 + intensity * 0.8})` 
                    : `rgba(244, 67, 54, ${0.1 + intensity * 0.6})`;
                
                tableHTML += `<td style="border: 1px solid #ddd; padding: 12px; text-align: center; background: ${color}; font-weight: ${i === j ? 'bold' : 'normal'}; min-width: 60px;">${val}</td>`;
            });
            
            tableHTML += '</tr>';
        });
        
        tableHTML += '</tbody></table>';
        
        // 添加说明
        tableHTML += '<p style="margin-top: 10px; font-size: 12px; color: #666; text-align: center;">';
        tableHTML += '<span style="display: inline-block; width: 15px; height: 15px; background: rgba(76, 175, 80, 0.6); margin-right: 5px; vertical-align: middle;"></span>正确预测 ';
        tableHTML += '<span style="display: inline-block; width: 15px; height: 15px; background: rgba(244, 67, 54, 0.6); margin-left: 15px; margin-right: 5px; vertical-align: middle;"></span>错误预测';
        tableHTML += '</p>';
        
        tableContainer.innerHTML = tableHTML;
        section.appendChild(tableContainer);
        container.appendChild(section);
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
     * 渲染多分类ROC曲线
     */
    renderMulticlassROCCurve(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>多分类ROC曲线 (One-vs-Rest)</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'roc-multiclass-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'];
        const datasets = [];
        
        Object.entries(data).forEach(([className, rocData], idx) => {
            datasets.push({
                label: `${className} (AUC=${rocData.auc.toFixed(3)})`,
                data: rocData.fpr.map((x, i) => ({ x, y: rocData.tpr[i] })),
                borderColor: colors[idx % colors.length],
                fill: false,
                tension: 0.1
            });
        });
        
        // 随机分类器基线
        datasets.push({
            label: '随机分类器',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#999',
            borderDash: [5, 5],
            fill: false
        });
        
        const chart = new Chart(canvas, {
            type: 'line',
            data: { datasets },
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
        
        this.charts.set('roc-multiclass', chart);
    }
    
    /**
     * 渲染错误分析
     */
    renderErrorAnalysis(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `
            <h3>错误分析</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">错误数</div>
                    <div class="metric-value">${data.total_errors}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">错误率</div>
                    <div class="metric-value">${(data.error_rate * 100).toFixed(2)}%</div>
                </div>
            </div>
        `;
        container.appendChild(section);
    }
    
    /**
     * 渲染回归可视化
     */
    renderRegressionVisualizations(container, visualizations) {
        // 预测vs实际
        if (visualizations.prediction_vs_actual) {
            this.renderPredictedVsActual(container, visualizations.prediction_vs_actual);
        }
        
        // 残差图
        if (visualizations.residual_plot) {
            this.renderResidualsPlot(container, visualizations.residual_plot);
        }
        
        // 残差分布
        if (visualizations.residual_distribution) {
            this.renderResidualDistribution(container, visualizations.residual_distribution);
        }
        
        // 误差分布
        if (visualizations.error_distribution) {
            this.renderErrorDistribution(container, visualizations.error_distribution);
        }
        
        // Q-Q图
        if (visualizations.qq_plot) {
            this.renderQQPlot(container, visualizations.qq_plot);
        }
        
        // 误差百分比分布
        if (visualizations.percentage_error_distribution) {
            this.renderPercentageErrorDistribution(container, visualizations.percentage_error_distribution);
        }
        
        // 预测区间分析
        if (visualizations.prediction_interval_analysis) {
            this.renderPredictionIntervalAnalysis(container, visualizations.prediction_interval_analysis);
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
     * 渲染残差分布
     */
    renderResidualDistribution(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>残差分布</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'residual-dist-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: data.bin_edges.slice(0, -1).map((e, i) => 
                    `${e.toFixed(2)} - ${data.bin_edges[i+1].toFixed(2)}`
                ),
                datasets: [{
                    label: '残差频数',
                    data: data.counts,
                    backgroundColor: '#2196F3'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { 
                        display: true, 
                        text: `均值: ${data.mean.toFixed(4)}, 标准差: ${data.std.toFixed(4)}` 
                    }
                },
                scales: {
                    x: { title: { display: true, text: '残差值' } },
                    y: { title: { display: true, text: '频数' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('residual-dist', chart);
    }
    
    /**
     * 渲染聚类可视化
     */
    renderClusteringVisualizations(container, visualizations) {
        // 2D投影
        if (visualizations['2d_projection']) {
            this.renderClusterScatter(container, visualizations['2d_projection']);
        }
        
        // 轮廓图
        if (visualizations.silhouette_plot) {
            this.renderSilhouettePlot(container, visualizations.silhouette_plot);
        }
        
        // 簇分布
        if (visualizations.cluster_distribution) {
            this.renderClusterDistribution(container, visualizations.cluster_distribution);
        }
        
        // 簇中心
        if (visualizations.cluster_centers) {
            this.renderClusterCenters(container, visualizations.cluster_centers);
        }
        
        // 簇间距离矩阵
        if (visualizations.inter_cluster_distances) {
            this.renderInterClusterDistances(container, visualizations.inter_cluster_distances);
        }
        
        // 簇特征统计
        if (visualizations.cluster_feature_stats) {
            this.renderClusterFeatureStats(container, visualizations.cluster_feature_stats);
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
     * 渲染聚簇分布
     */
    renderClusterDistribution(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>簇分布</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'cluster-dist-chart';
        canvas.height = 250;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const labels = Object.keys(data).map(k => `簇 ${k}`);
        const values = Object.values(data);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '样本数',
                    data: values,
                    backgroundColor: ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#FFEB3B']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { title: { display: true, text: '簇标签' } },
                    y: { title: { display: true, text: '样本数' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('cluster-dist', chart);
    }
    
    /**
     * 渲染轮廓图
     */
    renderSilhouettePlot(container, data) {
        // 验证数据是否存在
        if (!data || !data.silhouette_values || !Array.isArray(data.silhouette_values)) {
            console.warn('轮廓系数数据不存在或格式不正确:', data);
            return;
        }
        
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
                labels: data.sample_indices || Array.from({ length: data.silhouette_values.length }, (_, i) => i),
                datasets: [{
                    label: '轮廓系数',
                    data: data.silhouette_values,
                    backgroundColor: data.silhouette_values.map(v => 
                        v >= 0 ? `rgba(76, 175, 80, ${Math.abs(v)})` : `rgba(244, 67, 54, ${Math.abs(v)})`
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
        // 1. 2D散点图
        if (visualizations['2d_scatter']) {
            this.renderDimReductionScatter2D(container, visualizations['2d_scatter']);
        }
        
        // 2. 3D散点图（如果有）
        if (visualizations['3d_scatter']) {
            this.render3DScatter(container, visualizations['3d_scatter']);
        }
        
        // 3. 方差解释率（PCA等线性方法）
        if (visualizations['variance_explained']) {
            this.renderVarianceExplained(container, visualizations['variance_explained']);
        }
        
        // 4. 成分载荷图（PCA）
        if (visualizations['component_loadings']) {
            this.renderComponentLoadings(container, visualizations['component_loadings']);
        }
        
        // 5. 重构误差分布
        if (visualizations['reconstruction_error_distribution']) {
            this.renderReconstructionError(container, visualizations['reconstruction_error_distribution']);
        }
        
        // 6. 维度分布
        if (visualizations['dimension_distributions']) {
            this.renderDimensionDistributions(container, visualizations['dimension_distributions']);
        }
        
        // 7. 成分相关性矩阵
        if (visualizations['component_correlation']) {
            this.renderComponentCorrelation(container, visualizations['component_correlation']);
        }
        
        // 8. 特征值谱
        if (visualizations['eigenvalue_spectrum']) {
            this.renderEigenvalueSpectrum(container, visualizations['eigenvalue_spectrum']);
        }
    }
    
    /**
     * 渲染降维二维散点图
     */
    renderDimReductionScatter2D(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>降维结果可视化 (2D)</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'dimred-scatter-chart';
        canvas.height = 350;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '样本点',
                    data: data.x.map((x, i) => ({ x, y: data.y[i] })),
                    backgroundColor: 'rgba(33, 150, 243, 0.6)',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: (context) => `点 ${context.dataIndex}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: data.component_labels ? data.component_labels[0] : '维度 1' } },
                    y: { title: { display: true, text: data.component_labels ? data.component_labels[1] : '维度 2' } }
                }
            }
        });
        
        this.charts.set('dimred-scatter-2d', chart);
    }
    
    /**
     * 渲染方差解释率
     */
    renderVarianceExplained(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>方差解释率</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'variance-explained-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: data.components.map(c => `PC${c}`),
                datasets: [
                    {
                        label: '单个方差解释率',
                        data: data.variance_ratio.map(r => (r * 100).toFixed(2)),
                        backgroundColor: '#2196F3',
                        yAxisID: 'y'
                    },
                    {
                        label: '累积方差解释率',
                        data: data.cumulative_variance.map(c => (c * 100).toFixed(2)),
                        type: 'line',
                        borderColor: '#FF9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.2)',
                        yAxisID: 'y',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.parsed.y}%`
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: '主成分' } },
                    y: { 
                        title: { display: true, text: '方差解释率 (%)' },
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        this.charts.set('variance-explained', chart);
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
     * 渲染算法特定信息
     */
    renderAlgorithmSpecificInfo(container, info, algorithmName) {
        if (!info || Object.keys(info).length === 0) return;
        
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>算法特定信息 - ${info.algorithm || algorithmName}</h3>`;
        
        const infoContainer = document.createElement('div');
        infoContainer.style.cssText = 'background: #f9f9f9; padding: 15px; border-radius: 6px; margin-top: 10px;';
        
        // 模型参数
        if (info.model_params) {
            let paramsHTML = '<div style="margin-bottom: 15px;"><h4 style="margin-bottom: 10px; color: #333;">模型参数</h4>';
            paramsHTML += '<div class="metrics-grid">';
            
            Object.entries(info.model_params).forEach(([key, value]) => {
                paramsHTML += `
                    <div class="metric-card">
                        <div class="metric-label">${key}</div>
                        <div class="metric-value">${value}</div>
                    </div>
                `;
            });
            
            paramsHTML += '</div></div>';
            infoContainer.innerHTML += paramsHTML;
        }
        
        // CatBoost训练信息
        if (info.training_info) {
            let trainingHTML = '<div style="margin-bottom: 15px;"><h4 style="margin-bottom: 10px; color: #333;">训练信息</h4>';
            
            if (info.training_info.has_training_data) {
                trainingHTML += `<p style="color: #4CAF50; font-weight: bold;">✓ 训练数据已保存</p>`;
                trainingHTML += `<p style="font-size: 12px; color: #666;">路径: ${info.training_info.info_dir}</p>`;
            }
            
            if (info.training_info.final_metrics) {
                trainingHTML += `<p style="margin-top: 10px;"><strong>最终指标:</strong></p>`;
                trainingHTML += `<pre style="background: white; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto;">${info.training_info.final_metrics}</pre>`;
            }
            
            if (info.training_info.error) {
                trainingHTML += `<p style="color: #f44336;">错误: ${info.training_info.error}</p>`;
            }
            
            trainingHTML += '</div>';
            infoContainer.innerHTML += trainingHTML;
        }
        
        // SVM特定信息
        if (info.n_support_vectors) {
            let svmHTML = '<div style="margin-bottom: 15px;">';
            svmHTML += `<p><strong>支持向量数量:</strong> ${info.n_support_vectors}</p>`;
            svmHTML += '</div>';
            infoContainer.innerHTML += svmHTML;
        }
        
        // MLP训练曲线
        if (info.training_loss && Array.isArray(info.training_loss)) {
            let mlpHTML = '<div style="margin-bottom: 15px;"><h4 style="margin-bottom: 10px; color: #333;">训练损失曲线</h4>';
            
            const canvas = document.createElement('canvas');
            canvas.id = 'mlp-training-loss-chart';
            canvas.height = 200;
            
            infoContainer.innerHTML += mlpHTML;
            section.appendChild(infoContainer);
            section.appendChild(canvas);
            container.appendChild(section);
            
            // 绘制损失曲线
            const chart = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: info.training_loss.map((_, i) => i + 1),
                    datasets: [{
                        label: '训练损失',
                        data: info.training_loss,
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
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
                        x: { title: { display: true, text: '迭代次数' } },
                        y: { title: { display: true, text: '损失值' }, beginAtZero: true }
                    }
                }
            });
            
            this.charts.set('mlp-training-loss', chart);
            return;
        }
        
        section.appendChild(infoContainer);
        container.appendChild(section);
    }
    
    /**
     * 清空所有图表
     */
    clearAll() {
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
    }
    
    /**
     * 渲染3D散点图（使用数据点模拟）
     */
    render3DScatter(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>降维结果可视化 (3D)</h3>';
        
        // 使用三个2D投影模拟3D效果
        const viewsHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 10px;">
                <div>
                    <canvas id="3d-xy-chart" height="250"></canvas>
                </div>
                <div>
                    <canvas id="3d-xz-chart" height="250"></canvas>
                </div>
                <div>
                    <canvas id="3d-yz-chart" height="250"></canvas>
                </div>
            </div>
        `;
        section.innerHTML += viewsHTML;
        container.appendChild(section);
        
        // XY投影
        const xyChart = new Chart(document.getElementById('3d-xy-chart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'XY投影',
                    data: data.x.map((x, i) => ({ x, y: data.y[i] })),
                    backgroundColor: 'rgba(33, 150, 243, 0.6)',
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { title: { display: true, text: '维度 1' } },
                    y: { title: { display: true, text: '维度 2' } }
                }
            }
        });
        
        // XZ投影
        const xzChart = new Chart(document.getElementById('3d-xz-chart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'XZ投影',
                    data: data.x.map((x, i) => ({ x, y: data.z[i] })),
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { title: { display: true, text: '维度 1' } },
                    y: { title: { display: true, text: '维度 3' } }
                }
            }
        });
        
        // YZ投影
        const yzChart = new Chart(document.getElementById('3d-yz-chart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'YZ投影',
                    data: data.y.map((y, i) => ({ x: y, y: data.z[i] })),
                    backgroundColor: 'rgba(255, 152, 0, 0.6)',
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { title: { display: true, text: '维度 2' } },
                    y: { title: { display: true, text: '维度 3' } }
                }
            }
        });
        
        this.charts.set('3d-xy', xyChart);
        this.charts.set('3d-xz', xzChart);
        this.charts.set('3d-yz', yzChart);
    }
    
    /**
     * 渲染成分载荷图
     */
    renderComponentLoadings(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>主成分载荷矩阵</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'component-loadings-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const datasets = data.components.map((component, idx) => ({
            label: `PC${idx + 1}`,
            data: component,
            borderColor: `hsl(${idx * 120}, 70%, 50%)`,
            backgroundColor: `hsla(${idx * 120}, 70%, 50%, 0.1)`,
            tension: 0.1
        }));
        
        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.feature_indices.map(i => `F${i}`),
                datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { title: { display: true, text: '特征索引' } },
                    y: { title: { display: true, text: '载荷值' } }
                }
            }
        });
        
        this.charts.set('component-loadings', chart);
    }
    
    /**
     * 渲染重构误差分布
     */
    renderReconstructionError(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>重构误差分布</h3>
            <p style="color: #666; font-size: 14px;">平均误差: ${data.mean_error.toFixed(4)} | 中位数: ${data.median_error.toFixed(4)}</p>`;
        
        const canvas = document.createElement('canvas');
        canvas.id = 'reconstruction-error-chart';
        canvas.height = 280;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const binCenters = data.bin_edges.slice(0, -1).map((edge, i) => 
            (edge + data.bin_edges[i + 1]) / 2
        );
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: binCenters.map(c => c.toFixed(4)),
                datasets: [{
                    label: '样本数',
                    data: data.counts,
                    backgroundColor: '#9C27B0'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: '重构误差' } },
                    y: { title: { display: true, text: '样本数' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('reconstruction-error', chart);
    }
    
    /**
     * 渲染维度分布
     */
    renderDimensionDistributions(container, distributions) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>各维度数据分布</h3>';
        
        const chartsHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 10px;">
                ${distributions.map(dist => `
                    <div>
                        <canvas id="dim-dist-${dist.dimension}-chart" height="200"></canvas>
                        <p style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">
                            均值: ${dist.mean.toFixed(3)} | 标准差: ${dist.std.toFixed(3)}
                        </p>
                    </div>
                `).join('')}
            </div>
        `;
        section.innerHTML += chartsHTML;
        container.appendChild(section);
        
        distributions.forEach(dist => {
            const binCenters = dist.bin_edges.slice(0, -1).map((edge, i) => 
                (edge + dist.bin_edges[i + 1]) / 2
            );
            
            const chart = new Chart(document.getElementById(`dim-dist-${dist.dimension}-chart`), {
                type: 'bar',
                data: {
                    labels: binCenters.map(c => c.toFixed(2)),
                    datasets: [{
                        label: `维度 ${dist.dimension}`,
                        data: dist.counts,
                        backgroundColor: `hsl(${dist.dimension * 60}, 70%, 60%)`
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: true, position: 'top' } },
                    scales: {
                        x: { display: false },
                        y: { title: { display: true, text: '频数' }, beginAtZero: true }
                    }
                }
            });
            
            this.charts.set(`dim-dist-${dist.dimension}`, chart);
        });
    }
    
    /**
     * 渲染成分相关性矩阵
     */
    renderComponentCorrelation(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>成分相关性矩阵</h3>';
        
        // 使用HTML表格显示热力图
        const tableHTML = `
            <div style="overflow-x: auto; margin-top: 10px;">
                <table style="border-collapse: collapse; margin: 0 auto; background: white;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;"></th>
                            ${data.labels.map(label => `<th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">${label}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${data.matrix.map((row, i) => `
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">${data.labels[i]}</th>
                                ${row.map((val, j) => {
                                    const intensity = Math.abs(val);
                                    const color = val >= 0 
                                        ? `rgba(76, 175, 80, ${intensity})` 
                                        : `rgba(244, 67, 54, ${intensity})`;
                                    return `<td style="border: 1px solid #ddd; padding: 12px; text-align: center; background: ${color}; min-width: 60px;">${val.toFixed(3)}</td>`;
                                }).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        section.innerHTML += tableHTML;
        container.appendChild(section);
    }
    
    /**
     * 渲染特征值谱
     */
    renderEigenvalueSpectrum(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>特征值谱</h3>
            <p style="color: #666; font-size: 14px;">
                总方差: ${data.total_variance.toFixed(4)} | 
                前${data.eigenvalues.length}个特征值的方差: ${data.top_k_variance.toFixed(4)}
            </p>`;
        
        const canvas = document.createElement('canvas');
        canvas.id = 'eigenvalue-spectrum-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.eigenvalues.map((_, i) => i + 1),
                datasets: [{
                    label: '特征值',
                    data: data.eigenvalues,
                    borderColor: '#673AB7',
                    backgroundColor: 'rgba(103, 58, 183, 0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { title: { display: true, text: '特征值索引' } },
                    y: { title: { display: true, text: '特征值' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('eigenvalue-spectrum', chart);
    }
    
    /**
     * 渲染预测概率分布（分类）
     */
    renderProbabilityDistribution(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>预测概率分布（前100个样本）</h3>';
        
        // 创建表格显示概率
        const tableHTML = `
            <div style="overflow-x: auto; max-height: 400px; margin-top: 10px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead style="position: sticky; top: 0; background: #f5f5f5;">
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px;">样本</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">真实类别</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">预测类别</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">最高概率</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">所有类别概率</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.probabilities.map((probs, idx) => {
                            const maxProb = Math.max(...probs);
                            const isCorrect = data.true_classes[idx] === data.predicted_classes[idx];
                            const rowColor = isCorrect ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)';
                            return `
                                <tr style="background: ${rowColor};">
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${idx + 1}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${data.true_classes[idx]}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center; font-weight: bold;">${data.predicted_classes[idx]}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${(maxProb * 100).toFixed(1)}%</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; font-size: 11px;">${probs.map((p, i) => `C${i}:${(p*100).toFixed(1)}%`).join(' ')}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
        section.innerHTML += tableHTML;
        container.appendChild(section);
    }
    
    /**
     * 渲染类别预测分布（分类）
     */
    renderClassPredictionDistribution(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>类别预测分布</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'class-pred-dist-chart';
        canvas.height = 280;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const labels = Object.keys(data).map(k => `类别 ${k}`);
        const values = Object.values(data);
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: '预测样本数',
                    data: values,
                    backgroundColor: ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#FFEB3B']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: '类别' } },
                    y: { title: { display: true, text: '样本数' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('class-pred-dist', chart);
    }
    
    /**
     * 渲染误差分布（回归）
     */
    renderErrorDistribution(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>绝对误差分布</h3>
            <p style="color: #666; font-size: 14px;">平均误差: ${data.mean_error.toFixed(4)} | 中位数: ${data.median_error.toFixed(4)}</p>`;
        
        const canvas = document.createElement('canvas');
        canvas.id = 'error-dist-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const binCenters = data.bin_edges.slice(0, -1).map((e, i) => 
            (e + data.bin_edges[i+1]) / 2
        );
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: binCenters.map(c => c.toFixed(4)),
                datasets: [{
                    label: '频数',
                    data: data.counts,
                    backgroundColor: '#FF9800'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: '绝对误差' } },
                    y: { title: { display: true, text: '频数' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('error-dist', chart);
    }
    
    /**
     * 渲染Q-Q图（回归）
     */
    renderQQPlot(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>Q-Q图（残差正态性检验）</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'qq-plot-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const chart = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: '残差分位数',
                        data: data.theoretical_quantiles.map((x, i) => ({ 
                            x, 
                            y: data.sample_quantiles[i] 
                        })),
                        backgroundColor: 'rgba(156, 39, 176, 0.6)',
                        pointRadius: 4
                    },
                    {
                        label: '理论线',
                        data: [
                            { x: Math.min(...data.theoretical_quantiles), y: Math.min(...data.sample_quantiles) },
                            { x: Math.max(...data.theoretical_quantiles), y: Math.max(...data.sample_quantiles) }
                        ],
                        type: 'line',
                        borderColor: '#F44336',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'bottom' } },
                scales: {
                    x: { title: { display: true, text: '理论分位数' } },
                    y: { title: { display: true, text: '样本分位数' } }
                }
            }
        });
        
        this.charts.set('qq-plot', chart);
    }
    
    /**
     * 渲染误差百分比分布（回归）
     */
    renderPercentageErrorDistribution(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>误差百分比分布</h3>
            <p style="color: #666; font-size: 14px;">平均: ${data.mean_pct_error.toFixed(2)}% | 中位数: ${data.median_pct_error.toFixed(2)}%</p>`;
        
        const canvas = document.createElement('canvas');
        canvas.id = 'pct-error-dist-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const binCenters = data.bin_edges.slice(0, -1).map((e, i) => 
            (e + data.bin_edges[i+1]) / 2
        );
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: binCenters.map(c => c.toFixed(1) + '%'),
                datasets: [{
                    label: '频数',
                    data: data.counts,
                    backgroundColor: '#00BCD4'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: '误差百分比 (%)' } },
                    y: { title: { display: true, text: '频数' }, beginAtZero: true }
                }
            }
        });
        
        this.charts.set('pct-error-dist', chart);
    }
    
    /**
     * 渲染预测区间分析（回归）
     */
    renderPredictionIntervalAnalysis(container, binMetrics) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>预测区间分析</h3>';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'interval-analysis-chart';
        canvas.height = 300;
        section.appendChild(canvas);
        container.appendChild(section);
        
        const labels = binMetrics.map(m => 
            `[${m.bin_start.toFixed(2)}, ${m.bin_end.toFixed(2)}]`
        );
        
        const chart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'MAE',
                        data: binMetrics.map(m => m.mae),
                        backgroundColor: '#2196F3',
                        yAxisID: 'y'
                    },
                    {
                        label: 'R²',
                        data: binMetrics.map(m => m.r2 || 0),
                        type: 'line',
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.2)',
                        yAxisID: 'y1',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    x: { title: { display: true, text: '预测区间' } },
                    y: { 
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'MAE' },
                        beginAtZero: true
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'R²' },
                        min: -1,
                        max: 1,
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
        
        this.charts.set('interval-analysis', chart);
    }
    
    /**
     * 渲染簇中心（聚类）
     */
    renderClusterCenters(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>簇中心坐标</h3>';
        
        // 验证数据是否存在
        if (!data.centers || data.centers.length === 0) {
            section.innerHTML += '<p style="color: #999; padding: 20px; text-align: center;">该算法没有明确的簇中心（如DBSCAN）</p>';
            container.appendChild(section);
            return;
        }
        
        const tableHTML = `
            <div style="overflow-x: auto; margin-top: 10px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">簇</th>
                            ${Array.from({length: data.n_features}, (_, i) => 
                                `<th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">特征 ${i}</th>`
                            ).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${data.centers.map((center, idx) => `
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px; background: #f9f9f9;">簇 ${idx}</th>
                                ${center.map(val => 
                                    `<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">${val.toFixed(4)}</td>`
                                ).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        section.innerHTML += tableHTML;
        container.appendChild(section);
    }
    
    /**
     * 渲染簇间距离矩阵（聚类）
     */
    renderInterClusterDistances(container, data) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>簇间距离矩阵</h3>';
        
        // 验证数据是否存在
        if (!data.matrix || data.matrix.length === 0) {
            section.innerHTML += '<p style="color: #999; padding: 20px; text-align: center;">无法计算簇间距离（算法没有簇中心）</p>';
            container.appendChild(section);
            return;
        }
        
        const tableHTML = `
            <div style="overflow-x: auto; margin-top: 10px;">
                <table style="border-collapse: collapse; margin: 0 auto;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;"></th>
                            ${data.cluster_labels.map(label => 
                                `<th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">簇 ${label}</th>`
                            ).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${data.matrix.map((row, i) => `
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">簇 ${data.cluster_labels[i]}</th>
                                ${row.map((val, j) => {
                                    const maxDist = Math.max(...data.matrix.flat());
                                    const intensity = maxDist > 0 ? val / maxDist : 0;
                                    const color = i === j 
                                        ? `rgba(200, 200, 200, 0.5)` 
                                        : `rgba(33, 150, 243, ${0.2 + intensity * 0.6})`;
                                    return `<td style="border: 1px solid #ddd; padding: 10px; text-align: center; background: ${color}; min-width: 60px;">${val.toFixed(3)}</td>`;
                                }).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        section.innerHTML += tableHTML;
        container.appendChild(section);
    }
    
    /**
     * 渲染簇特征统计（聚类）
     */
    renderClusterFeatureStats(container, clusterStats) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = '<h3>簇特征统计</h3>';
        
        // 验证数据是否存在
        const clusters = Object.keys(clusterStats);
        if (clusters.length === 0 || !clusterStats[clusters[0]]) {
            section.innerHTML += '<p style="color: #999; padding: 20px; text-align: center;">无有效簇数据（可能所有点都被标记为噪声点）</p>';
            container.appendChild(section);
            return;
        }
        
        const firstCluster = clusterStats[clusters[0]];
        const nFeatures = firstCluster.mean.length;
        
        // 限制显示的特征数量
        const maxFeatures = Math.min(10, nFeatures);
        
        const tableHTML = `
            <div style="overflow-x: auto; margin-top: 10px;">
                <p style="color: #666; font-size: 13px; margin-bottom: 10px;">
                    显示前 ${maxFeatures} 个特征的统计信息（总共 ${nFeatures} 个特征）
                </p>
                ${clusters.map(cluster => `
                    <h4 style="margin-top: 15px; color: #333;">簇 ${cluster}</h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 15px;">
                        <thead>
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 6px; background: #f5f5f5;">特征</th>
                                <th style="border: 1px solid #ddd; padding: 6px; background: #f5f5f5;">均值</th>
                                <th style="border: 1px solid #ddd; padding: 6px; background: #f5f5f5;">标准差</th>
                                <th style="border: 1px solid #ddd; padding: 6px; background: #f5f5f5;">最小值</th>
                                <th style="border: 1px solid #ddd; padding: 6px; background: #f5f5f5;">最大值</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Array.from({length: maxFeatures}, (_, i) => `
                                <tr>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">特征 ${i}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${clusterStats[cluster].mean[i].toFixed(4)}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${clusterStats[cluster].std[i].toFixed(4)}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${clusterStats[cluster].min[i].toFixed(4)}</td>
                                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">${clusterStats[cluster].max[i].toFixed(4)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `).join('')}
            </div>
        `;
        section.innerHTML += tableHTML;
        container.appendChild(section);
    }
}

// 导出
window.ResultVisualizer = ResultVisualizer;
