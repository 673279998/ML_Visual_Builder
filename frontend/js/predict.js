/**
 * 模型预测功能
 */

class PredictManager {
    constructor() {
        this.models = [];
        this.datasets = [];
        this.selectedModel = null;
        this.init();
    }

    async init() {
        this.bindEvents();
        await this.loadModels();
        await this.loadDatasets();
        await this.loadPredictionHistory();
    }

    bindEvents() {
        // 模型选择
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.onModelSelect(e.target.value);
        });

        // 输入方式切换
        document.getElementById('input-mode').addEventListener('change', (e) => {
            this.toggleInputMode(e.target.value);
        });

        // 预测按钮
        document.getElementById('predict-btn').addEventListener('click', () => {
            this.predict();
        });

        // 清空按钮
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearInputs();
        });
    }

    async loadModels() {
        try {
            const result = await apiClient.getModels();
            
            if (result.success) {
                this.models = result.data;
                this.renderModelOptions();
            } else {
                UIHelper.showMessage('加载模型列表失败: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('加载模型失败:', error);
            UIHelper.showMessage('加载模型列表失败', 'error');
        }
    }

    async loadDatasets() {
        try {
            const result = await apiClient.getDatasets();
            
            if (result.success) {
                this.datasets = result.data;
                this.renderDatasetOptions();
            }
        } catch (error) {
            console.error('加载数据集失败:', error);
        }
    }

    renderModelOptions() {
        const select = document.getElementById('model-select');
        select.innerHTML = '<option value="">请选择模型...</option>';
        
        this.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            const algorithmName = model.algorithm_display_name || model.algorithm_name;
            option.textContent = `${model.name} (${algorithmName})`;
            select.appendChild(option);
        });
    }

    renderDatasetOptions() {
        const select = document.getElementById('dataset-select');
        select.innerHTML = '<option value="">请选择数据集...</option>';
        
        this.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = `${dataset.name} (${dataset.row_count}行)`;
            select.appendChild(option);
        });
    }

    async onModelSelect(modelId) {
        if (!modelId) {
            document.getElementById('model-info').style.display = 'none';
            this.selectedModel = null;
            return;
        }

        try {
            // 获取模型详细信息
            const result = await apiClient.getModel(modelId);
            if (result.success) {
                this.selectedModel = result.data;
                this.displayModelInfo(this.selectedModel);
            } else {
                throw new Error(result.error || '获取模型详情失败');
            }
        } catch (error) {
            console.error('获取模型详情失败:', error);
            UIHelper.showMessage('获取模型详情失败: ' + error.message, 'error');
        }
    }

    displayModelInfo(model) {
        const infoDiv = document.getElementById('model-info');
        
        document.getElementById('info-algorithm-type').textContent = model.algorithm_type;
        const algorithmName = model.algorithm_display_name || model.algorithm_name;
        document.getElementById('info-algorithm-name').textContent = algorithmName;
        document.getElementById('info-created-at').textContent = new Date(model.created_at).toLocaleString('zh-CN');
        
        // 显示输入特征
        const features = model.input_requirements?.features || model.input_requirements?.required_columns || [];
        document.getElementById('info-features').textContent = features.join(', ') || '无';
        
        // 显示预处理详情
        this.displayPreprocessingInfo(model);
        
        infoDiv.style.display = 'block';

        // 更新输入提示
        if (features.length > 0) {
            const exampleData = {};
            features.forEach(f => {
                exampleData[f] = 0.0;
            });
            const exampleJson = JSON.stringify([exampleData], null, 2);
            document.getElementById('input-data').placeholder = 
                `请输入JSON格式的数据，例如:\n${exampleJson}`;
        }
    }

    displayPreprocessingInfo(model) {
        const container = document.getElementById('preprocessing-info');
        if (!container) return;
        
        container.style.display = 'block';
        
        // 1. 数据集实例格式
        const features = model.input_requirements?.features || model.input_requirements?.required_columns || [];
        const formatExample = {};
        features.forEach(f => formatExample[f] = "<数值/类别>");
        document.getElementById('info-dataset-format').textContent = JSON.stringify(formatExample, null, 2);
        
        // 2. 特征列信息 (编码方式等)
        const columnsDiv = document.getElementById('info-columns');
        const components = model.preprocessing_components || [];
        
        // 整理列信息
        let columnsHtml = '<table style="width:100%; border-collapse: collapse; font-size: 14px;">' + 
            '<thead><tr style="background-color: #f8f9fa;">' + 
            '<th style="text-align:left; padding: 8px; border-bottom:2px solid #dee2e6;">列名</th>' + 
            '<th style="text-align:left; padding: 8px; border-bottom:2px solid #dee2e6;">类型</th>' + 
            '<th style="text-align:left; padding: 8px; border-bottom:2px solid #dee2e6;">处理流程 (填充 → 编码 → 缩放)</th>' + 
            '</tr></thead><tbody>';
        
        if (features.length === 0) {
            columnsHtml += '<tr><td colspan="3" style="padding:8px; text-align:center; color:#6c757d;">无特征信息</td></tr>';
        } else {
            features.forEach(feature => {
                let methods = [];
                let type = '数值';
                
                // 查找该特征涉及的预处理组件
                const relatedComps = components.filter(c => {
                    // 使用applied_columns字段判断
                    if (c.applied_columns && c.applied_columns.includes(feature)) return true;
                    return false;
                });
                
                // 按照处理顺序排序: imputer -> encoder -> scaler
                const typeOrder = { 'imputer': 1, 'encoder': 2, 'scaler': 3, 'outlier': 4 };
                relatedComps.sort((a, b) => {
                    const typeA = a.type || a.component_type;
                    const typeB = b.type || b.component_type;
                    return (typeOrder[typeA] || 99) - (typeOrder[typeB] || 99);
                });

                if (relatedComps.length > 0) {
                    relatedComps.forEach(c => {
                        const config = c.configuration || {};
                        const method = config.method || c.method || config.strategy || 'Unknown';
                        const cType = c.type || c.component_type;
                        
                        if (cType === 'imputer') {
                            methods.push(`<span style="color:#0d6efd;">填充</span>(${method})`);
                        } else if (cType === 'encoder') {
                            methods.push(`<span style="color:#d63384;">编码</span>(${method})`);
                            type = '类别'; // 只要有编码，推断为类别型
                        } else if (cType === 'scaler') {
                            methods.push(`<span style="color:#198754;">缩放</span>(${method})`);
                        } else if (cType === 'outlier') {
                            methods.push(`<span style="color:#fd7e14;">异常值</span>(${method})`);
                        } else {
                            methods.push(`${cType}(${method})`);
                        }
                    });
                }
                
                const processStr = methods.length > 0 ? methods.join(' → ') : '<span style="color:#6c757d;">原值使用</span>';
                
                columnsHtml += `<tr style="border-bottom:1px solid #eee;">` + 
                    `<td style="padding:8px;">${feature}</td>` + 
                    `<td style="padding:8px;">${type}</td>` + 
                    `<td style="padding:8px;">${processStr}</td>` + 
                    `</tr>`;
            });
        }
        columnsHtml += '</tbody></table>';
        columnsDiv.innerHTML = columnsHtml;
        
        // 3. 预处理方法汇总
        const componentsDiv = document.getElementById('info-components');
        if (components.length === 0) {
            componentsDiv.innerHTML = '<div style="color:#6c757d; font-style:italic;">无预处理组件</div>';
        } else {
            const compList = components.map(c => {
                let desc = '';
                const config = c.configuration || {};
                const method = config.method || c.method || config.strategy || 'Standard';
                const columnsStr = c.applied_columns?.join(', ') || '全部';
                const type = c.type || c.component_type;
                
                switch(type) {
                    case 'encoder': desc = `<strong>编码器</strong>: ${method} <br><span style="font-size:0.9em;color:#666;">应用列: ${columnsStr}</span>`; break;
                    case 'scaler': desc = `<strong>缩放器</strong>: ${method} <br><span style="font-size:0.9em;color:#666;">应用列: ${columnsStr}</span>`; break;
                    case 'imputer': desc = `<strong>填充器</strong>: ${method} <br><span style="font-size:0.9em;color:#666;">应用列: ${columnsStr}</span>`; break;
                    case 'outlier': desc = `<strong>异常值处理</strong>: ${method}`; break;
                    default: desc = `<strong>${type}</strong>: ${method}`;
                }
                return `<div style="margin-bottom:8px; padding:4px; background:#f8f9fa; border-radius:4px;">• ${desc}</div>`;
            }).join('');
            componentsDiv.innerHTML = compList;
        }
    }

    toggleInputMode(mode) {
        const manualInput = document.getElementById('manual-input');
        const datasetInput = document.getElementById('dataset-input');
        
        if (mode === 'manual') {
            manualInput.style.display = 'block';
            datasetInput.style.display = 'none';
        } else {
            manualInput.style.display = 'none';
            datasetInput.style.display = 'block';
        }
    }

    async predict() {
        if (!this.selectedModel) {
            UIHelper.showMessage('请先选择模型', 'error');
            return;
        }

        const inputMode = document.getElementById('input-mode').value;
        const returnProbabilities = document.getElementById('return-probabilities').checked;
        
        let inputData;
        
        try {
            if (inputMode === 'manual') {
                // 手动输入
                const inputText = document.getElementById('input-data').value.trim();
                if (!inputText) {
                    UIHelper.showMessage('请输入预测数据', 'error');
                    return;
                }
                
                inputData = JSON.parse(inputText);
                if (!Array.isArray(inputData)) {
                    UIHelper.showMessage('输入数据必须是数组格式', 'error');
                    return;
                }
                
                await this.predictWithData(inputData, returnProbabilities);
                
            } else {
                // 数据集预测
                const datasetId = document.getElementById('dataset-select').value;
                if (!datasetId) {
                    UIHelper.showMessage('请选择数据集', 'error');
                    return;
                }
                
                await this.predictWithDataset(datasetId);
            }
            
        } catch (error) {
            console.error('预测失败:', error);
            UIHelper.showMessage('预测失败: ' + error.message, 'error');
        }
    }

    async predictWithData(inputData, returnProbabilities) {
        // 显示加载
        document.getElementById('loading').classList.add('active');
        document.getElementById('results-section').style.display = 'none';
        
        try {
            const result = await apiClient.predict({
                model_id: this.selectedModel.id,
                input_data: inputData,
                return_probabilities: returnProbabilities
            });
            
            if (result.success) {
                this.displayResults(result.data);
                await this.loadPredictionHistory();
            } else {
                UIHelper.showMessage('预测失败: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('预测请求失败:', error);
            UIHelper.showMessage('预测请求失败', 'error');
        } finally {
            document.getElementById('loading').classList.remove('active');
        }
    }

    async predictWithDataset(datasetId) {
        // 显示加载
        document.getElementById('loading').classList.add('active');
        document.getElementById('results-section').style.display = 'none';
        
        try {
            const result = await apiClient.batchPredict({
                model_id: this.selectedModel.id,
                dataset_id: datasetId
            });
            
            if (result.success) {
                this.displayResults(result.data);
                await this.loadPredictionHistory();
            } else {
                UIHelper.showMessage('批量预测失败: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('批量预测失败:', error);
            UIHelper.showMessage('批量预测请求失败', 'error');
        } finally {
            document.getElementById('loading').classList.remove('active');
        }
    }

    displayResults(data) {
        const resultsSection = document.getElementById('results-section');
        const resultsContent = document.getElementById('results-content');
        
        UIHelper.showMessage(`预测完成！共预测 ${data.n_samples} 条数据`, 'success');
        
        let html = '<table class="results-table"><thead><tr>';
        html += '<th>序号</th>';
        html += '<th>预测结果</th>';
        
        // 如果有概率信息
        if (data.probabilities && data.class_labels) {
            data.class_labels.forEach(label => {
                html += `<th>类别 ${label} 概率</th>`;
            });
        }
        
        html += '</tr></thead><tbody>';
        
        data.predictions.forEach((pred, index) => {
            html += '<tr>';
            html += `<td>${index + 1}</td>`;
            html += `<td><strong>${pred}</strong></td>`;
            
            // 显示概率
            if (data.probabilities && data.probabilities[index]) {
                data.probabilities[index].forEach(prob => {
                    const percentage = (prob * 100).toFixed(2);
                    html += `<td>
                        <div class="probability-bar" style="width: ${percentage}%">
                            <span class="probability-value">${percentage}%</span>
                        </div>
                    </td>`;
                });
            }
            
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        
        resultsContent.innerHTML = html;
        resultsSection.style.display = 'block';
        
        // 滚动到结果区域
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async loadPredictionHistory() {
        try {
            const result = await apiClient.getPredictions({ limit: 10 });
            
            if (result.success) {
                this.renderPredictionHistory(result.data);
            }
        } catch (error) {
            console.error('加载预测历史失败:', error);
        }
    }

    renderPredictionHistory(predictions) {
        const historyList = document.getElementById('history-list');
        
        if (predictions.length === 0) {
            historyList.innerHTML = '<p style="color: #999; text-align: center;">暂无预测历史</p>';
            return;
        }
        
        historyList.innerHTML = '';
        
        predictions.forEach(pred => {
            const model = this.models.find(m => m.id === pred.model_id);
            const modelName = model ? model.name : `模型 ${pred.model_id}`;
            
            const summary = pred.prediction_summary || {};
            const nSamples = summary.n_samples || 0;
            
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-item-header">
                    <strong>${modelName}</strong>
                    <span class="history-item-time">${new Date(pred.created_at).toLocaleString('zh-CN')}</span>
                </div>
                <div class="history-item-info">
                    预测了 ${nSamples} 条数据
                </div>
            `;
            
            item.addEventListener('click', () => {
                this.showPredictionDetail(pred);
            });
            
            historyList.appendChild(item);
        });
    }

    showPredictionDetail(prediction) {
        const summary = prediction.prediction_summary || {};
        
        if (summary.predictions) {
            const data = {
                predictions: summary.predictions,
                probabilities: summary.probabilities,
                class_labels: summary.class_labels,
                n_samples: summary.n_samples
            };
            
            this.displayResults(data);
        } else {
            UIHelper.showMessage('该预测记录没有详细结果信息', 'info');
        }
    }

    clearInputs() {
        document.getElementById('input-data').value = '';
        document.getElementById('dataset-select').value = '';
        document.getElementById('return-probabilities').checked = false;
        document.getElementById('results-section').style.display = 'none';
        document.getElementById('alert-container').innerHTML = '';
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new PredictManager();
});
