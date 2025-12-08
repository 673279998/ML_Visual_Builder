/**
 * 工作流管理器 - 负责工作流的执行和与后端交互
 */

class WorkflowManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.currentWorkflowId = null;
        this.isExecuting = false;
        
        this.init();
    }
    
    init() {
        // 监听节点配置事件
        document.addEventListener('node-config', (e) => {
            this.showNodeConfigModal(e.detail);
        });
        
        // 绑定执行按钮
        const executeBtn = document.getElementById('btn-execute');
        if (executeBtn) {
            executeBtn.addEventListener('click', () => this.executeWorkflow());
        }
        
        // 绑定保存按钮
        const saveBtn = document.getElementById('btn-save');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveWorkflow());
        }

        // 绑定管理工作流按钮
        const manageBtn = document.getElementById('btn-manage-workflows');
        if (manageBtn) {
            manageBtn.addEventListener('click', () => this.showWorkflowManager());
        }

        // 绑定工作流管理对话框事件
        const closeManageBtn = document.getElementById('btn-close-workflow-manage');
        if (closeManageBtn) {
            closeManageBtn.addEventListener('click', () => this.hideWorkflowManager());
        }

        const refreshWorkflowsBtn = document.getElementById('btn-refresh-workflows');
        if (refreshWorkflowsBtn) {
            refreshWorkflowsBtn.addEventListener('click', () => this.loadWorkflows());
        }

        const batchDeleteWorkflowsBtn = document.getElementById('btn-batch-delete-workflows');
        if (batchDeleteWorkflowsBtn) {
            batchDeleteWorkflowsBtn.addEventListener('click', () => this.batchDeleteWorkflows());
        }
    }
    
    /**
     * 显示节点配置对话框
     */
    showNodeConfigModal(detail) {
        const { nodeId, nodeName, config } = detail;
        
        // 根据节点类型显示不同的配置界面
        switch(nodeName) {
            case '数据上传':
                this.showDataUploadConfig(nodeId, config);
                break;
            case '数据集选择':
                this.showDatasetConfig(nodeId, config);
                break;
            case '预处理':
                this.showPreprocessingConfig(nodeId, config);
                break;
            case '异常值检测':
                this.showOutlierDetectionConfig(nodeId, config);
                break;
            case '特征选择':
                this.showFeatureSelectionConfig(nodeId, config);
                break;
            case '分类算法选择':
                this.showAlgorithmConfig(nodeId, 'classification', config);
                break;
            case '回归算法选择':
                this.showAlgorithmConfig(nodeId, 'regression', config);
                break;
            case '聚类算法选择':
                this.showAlgorithmConfig(nodeId, 'clustering', config);
                break;
            case '降维算法选择':
                this.showAlgorithmConfig(nodeId, 'dimensionality_reduction', config);
                break;
            case '超参数调优':
                this.showHyperparameterTuningConfig(nodeId, config);
                break;
            case '模型训练':
                this.showModelTrainingConfig(nodeId, config);
                break;
            case '模型结果':
            case '可视化':
            case '终止':
                this.showOutputModuleConfig(nodeId, nodeName, config);
                break;
            default:
                this.showGenericConfig(nodeId, nodeName, config);
        }
    }
    
    /**
     * 显示数据上传配置
     */
    showDataUploadConfig(nodeId, config) {
        this.showModal('数据上传配置', `
            <div class="form-group">
                <label>上传数据集文件:</label>
                <div class="input-group">
                    <input type="file" id="config-upload-file" class="form-control" accept=".csv,.xlsx,.xls,.json" style="display:none">
                    <button class="btn btn-outline-secondary" type="button" onclick="document.getElementById('config-upload-file').click()">选择文件</button>
                    <input type="text" id="file-name-display" class="form-control" readonly placeholder="未选择文件" onclick="document.getElementById('config-upload-file').click()" style="cursor:pointer">
                </div>
                <small class="form-text text-muted">支持 CSV, Excel, JSON 格式</small>
            </div>
            <div class="form-group" style="text-align: center;">
                 <button id="btn-upload-action" class="btn btn-success" disabled>立即上传</button>
            </div>
            <div class="form-group" id="upload-progress-container" style="display: none;">
                <label>上传进度:</label>
                <div class="progress">
                    <div id="upload-progress-bar" class="progress-bar bg-info" role="progressbar" style="width: 0%">0%</div>
                </div>
                <small id="upload-status-text" class="form-text text-muted"></small>
            </div>
            <div class="form-group">
                <label>当前已选数据集:</label>
                <input type="text" id="config-dataset-name" class="form-control" readonly 
                       value="${config.dataset_name || '未选择'}">
                <input type="hidden" id="config-dataset-id" value="${config.dataset_id || ''}">
            </div>
        `, () => {
            const datasetId = document.getElementById('config-dataset-id').value;
            const datasetName = document.getElementById('config-dataset-name').value;
            
            if (datasetId) {
                this.canvas.updateNodeConfig(nodeId, { 
                    dataset_id: parseInt(datasetId),
                    dataset_name: datasetName
                });
                this.hideModal();
            } else {
                alert('请先上传并成功保存数据集');
            }
        });

        // 绑定文件选择显示
        const fileInput = document.getElementById('config-upload-file');
        const fileNameDisplay = document.getElementById('file-name-display');
        const uploadBtn = document.getElementById('btn-upload-action');
        
        if (fileInput && fileNameDisplay && uploadBtn) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    fileNameDisplay.value = e.target.files[0].name;
                    uploadBtn.disabled = false;
                } else {
                    fileNameDisplay.value = '';
                    uploadBtn.disabled = true;
                }
            });

            // 绑定上传按钮事件
            uploadBtn.addEventListener('click', () => {
                const file = fileInput.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                const progressContainer = document.getElementById('upload-progress-container');
                const progressBar = document.getElementById('upload-progress-bar');
                const statusText = document.getElementById('upload-status-text');
                
                progressContainer.style.display = 'block';
                progressBar.style.width = '0%';
                progressBar.textContent = '0%';
                progressBar.className = 'progress-bar bg-info';
                statusText.textContent = '正在上传...';
                uploadBtn.disabled = true;

                // 使用 XMLHttpRequest 来获取上传进度
                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/api/data/upload');
                
                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = Math.round((e.loaded / e.total) * 100);
                        progressBar.style.width = percentComplete + '%';
                        progressBar.textContent = percentComplete + '%';
                    }
                };
                
                xhr.onload = () => {
                    if (xhr.status === 200) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            if (response.success) {
                                progressBar.classList.remove('bg-info');
                                progressBar.classList.add('bg-success');
                                statusText.textContent = '上传成功！';
                                
                                // 更新隐藏域和显示
                                document.getElementById('config-dataset-id').value = response.data.dataset_id;
                                document.getElementById('config-dataset-name').value = response.data.name;
                                
                            } else {
                                progressBar.classList.remove('bg-info');
                                progressBar.classList.add('bg-danger');
                                statusText.textContent = '上传失败: ' + (response.error || '未知错误');
                                uploadBtn.disabled = false;
                            }
                        } catch (e) {
                            progressBar.classList.remove('bg-info');
                            progressBar.classList.add('bg-danger');
                            statusText.textContent = '解析响应失败';
                            uploadBtn.disabled = false;
                        }
                    } else {
                        progressBar.classList.remove('bg-info');
                        progressBar.classList.add('bg-danger');
                        statusText.textContent = '上传失败: 服务器错误 ' + xhr.status;
                        uploadBtn.disabled = false;
                    }
                };
                
                xhr.onerror = () => {
                    progressBar.classList.remove('bg-info');
                    progressBar.classList.add('bg-danger');
                    statusText.textContent = '网络错误';
                    uploadBtn.disabled = false;
                };
                
                xhr.send(formData);
            });
        }
    }
    
    /**
     * 显示数据集选择配置
     */
    async showDatasetConfig(nodeId, config) {
        try {
            const response = await fetch('/api/data/datasets');
            const result = await response.json();
            
            if (!result.success) {
                throw new Error('加载数据集列表失败');
            }
            
            const datasets = result.data || [];
            
            const options = datasets.map(ds => 
                `<option value="${ds.id}" ${config.dataset_id == ds.id ? 'selected' : ''}>${ds.name}</option>`
            ).join('');
            
            this.showModal('数据集配置', `
                <div class="form-group">
                    <label>选择数据集:</label>
                    <select id="config-dataset-id" class="form-control">
                        <option value="">请选择...</option>
                        ${options}
                    </select>
                </div>
            `, () => {
                const select = document.getElementById('config-dataset-id');
                const datasetId = select.value;
                if (datasetId) {
                    const datasetName = select.options[select.selectedIndex].text;
                    this.canvas.updateNodeConfig(nodeId, { 
                        dataset_id: parseInt(datasetId),
                        dataset_name: datasetName
                    });
                    this.hideModal();
                } else {
                    alert('请选择数据集');
                }
            });
        } catch (error) {
            console.error('加载数据集失败:', error);
            alert('加载数据集失败: ' + error.message);
        }
    }
    
    /**
     * 显示预处理配置
     */
    showPreprocessingConfig(nodeId, config) {
        this.showModal('预处理配置', `
            <div class="form-group">
                <label>
                    <input type="checkbox" id="config-handle-missing" 
                           ${config.handle_missing ? 'checked' : ''}>
                    处理缺失值
                </label>
            </div>
            <div class="form-group">
                <label>缺失值策略:</label>
                <select id="config-missing-strategy" class="form-control">
                    <option value="mean" ${config.missing_strategy === 'mean' ? 'selected' : ''}>均值填充</option>
                    <option value="median" ${config.missing_strategy === 'median' ? 'selected' : ''}>中位数填充</option>
                    <option value="mode" ${config.missing_strategy === 'mode' ? 'selected' : ''}>众数填充</option>
                    <option value="knn" ${config.missing_strategy === 'knn' ? 'selected' : ''}>K近邻填充</option>
                    <option value="tree" ${config.missing_strategy === 'tree' ? 'selected' : ''}>树模型填充</option>
                    <option value="drop" ${config.missing_strategy === 'drop' ? 'selected' : ''}>删除</option>
                </select>
            </div>
            <div class="form-group">
                <label>
                    <input type="checkbox" id="config-scale" 
                           ${config.scale ? 'checked' : ''}>
                    数据缩放
                </label>
            </div>
            <div class="form-group">
                <label>缩放方法:</label>
                <select id="config-scale-method" class="form-control">
                    <option value="standard" ${config.scale_method === 'standard' ? 'selected' : ''}>标准化</option>
                    <option value="minmax" ${config.scale_method === 'minmax' ? 'selected' : ''}>归一化</option>
                    <option value="robust" ${config.scale_method === 'robust' ? 'selected' : ''}>鲁棒缩放</option>
                </select>
            </div>
        `, () => {
            this.canvas.updateNodeConfig(nodeId, {
                handle_missing: document.getElementById('config-handle-missing').checked,
                missing_strategy: document.getElementById('config-missing-strategy').value,
                scale: document.getElementById('config-scale').checked,
                scale_method: document.getElementById('config-scale-method').value
            });
            this.hideModal();
        });
    }
    
    /**
     * 显示模型训练配置
     */
    showModelTrainingConfig(nodeId, config) {
        this.showModal('模型训练配置', `
            <div class="form-group">
                <label>目标列 (留空则使用数据集默认目标):</label>
                <input type="text" id="config-target-columns" class="form-control" 
                       value="${(config.target_columns || []).join(',')}" 
                       placeholder="例如: target">
                <small class="form-text text-muted">多个目标列用逗号分隔</small>
            </div>
            <div class="form-group">
                <label>测试集比例:</label>
                <input type="number" id="config-test-size" class="form-control" 
                       value="${config.test_size || 0.2}" 
                       min="0.1" max="0.5" step="0.05">
            </div>
             <div class="form-group">
                <label>
                    <input type="checkbox" id="config-random-state" 
                           ${config.use_random_state !== false ? 'checked' : ''}>
                    固定随机种子 (42)
                </label>
            </div>
        `, () => {
            const targetColumns = document.getElementById('config-target-columns').value
                .split(',').map(c => c.trim()).filter(c => c);
            const testSize = parseFloat(document.getElementById('config-test-size').value);
            
            this.canvas.updateNodeConfig(nodeId, {
                target_columns: targetColumns,
                test_size: testSize,
                use_random_state: document.getElementById('config-random-state').checked
            });
            this.hideModal();
        });
    }

    /**
     * 显示算法配置
     */
    async showAlgorithmConfig(nodeId, algorithmType, config) {
        try {
            const response = await fetch('/api/algorithms');
            const result = await response.json();
            
            if (!result.success) {
                throw new Error('加载算法列表失败');
            }
            
            const algorithms = result.data[algorithmType] || [];
            
            // 兼容两种数据结构：数组或对象
            let targetAlgorithms = algorithms;
            if (!Array.isArray(algorithms) && typeof algorithms === 'object') {
                 // 如果返回的是全量数据（按类型分组），尝试获取对应类型
                 targetAlgorithms = algorithms[algorithmType] || [];
            } else if (result.data && result.data[algorithmType]) {
                 targetAlgorithms = result.data[algorithmType];
            } else if (Array.isArray(result.data)) {
                 // 如果是扁平数组，过滤出对应类型
                 targetAlgorithms = result.data.filter(alg => alg.category === algorithmType || alg.type === algorithmType);
            }
            
            let algorithmOptions = targetAlgorithms.map(alg =>
                `<option value="${alg.name}" ${config.algorithm === alg.name ? 'selected' : ''}>${alg.display_name}</option>`
            ).join('');
            
            this.showModal('算法选择配置', `
                <div class="form-group">
                    <label>选择算法:</label>
                    <select id="config-algorithm" class="form-control">
                        <option value="">请选择...</option>
                        ${algorithmOptions}
                    </select>
                </div>
                <div class="form-group">
                    <label>固定超参数 (JSON格式):</label>
                    <textarea id="config-hyperparameters" class="form-control" rows="4" 
                              placeholder='例如: {"random_state": 42, "n_jobs": -1}'>${JSON.stringify(config.fixed_hyperparameters || config.hyperparameters || {}, null, 2)}</textarea>
                    <small class="form-text text-muted">在此处设置不需要调优的固定参数。如需调优，请在"超参数调优"节点中设置。</small>
                </div>
            `, () => {
                const algorithm = document.getElementById('config-algorithm').value;
                const hyperparametersText = document.getElementById('config-hyperparameters').value.trim();
                
                if (!algorithm) {
                    alert('请选择算法');
                    return;
                }
                
                let hyperparameters = {};
                if (hyperparametersText) {
                    try {
                        hyperparameters = JSON.parse(hyperparametersText);
                    } catch (e) {
                        alert('超参数JSON格式错误');
                        return;
                    }
                }
                
                this.canvas.updateNodeConfig(nodeId, {
                    algorithm: algorithm,
                    fixed_hyperparameters: hyperparameters, // 重命名为固定参数
                    algorithm_type: algorithmType
                });
                this.hideModal();
            });
        } catch (error) {
            console.error('加载算法列表失败:', error);
            alert('加载算法列表失败: ' + error.message);
        }
    }
    
    /**
     * 显示通用配置
     */
    showGenericConfig(nodeId, nodeName, config) {
        this.showModal(`${nodeName} 配置`, `
            <p>该节点暂无特殊配置项</p>
        `, () => {
            this.hideModal();
        });
    }
    
    /**
     * 显示超参数调优配置
     */
    showHyperparameterTuningConfig(nodeId, config) {
        this.showModal('超参数调优配置', `
            <div class="form-group">
                <label>调优方法:</label>
                <select id="config-tuning-method" class="form-control">
                    <option value="grid" ${config.method === 'grid' ? 'selected' : ''}>网格搜索</option>
                    <option value="random" ${config.method === 'random' ? 'selected' : ''}>随机搜索</option>
                    <option value="bayesian" ${config.method === 'bayesian' ? 'selected' : ''}>贝叶斯优化</option>
                </select>
            </div>
            <div class="form-group">
                <label>交叉验证折数:</label>
                <input type="number" id="config-cv-folds" class="form-control" 
                       value="${config.cv || 5}" min="2" max="10">
            </div>
            <div class="form-group">
                <label>迭代次数 (网格搜索无效):</label>
                <input type="number" id="config-n-iter" class="form-control" 
                       value="${config.n_iter || 100}" min="10" max="500">
            </div>
            <div class="form-group">
                <label>评估指标:</label>
                <select id="config-scoring" class="form-control">
                    <option value="accuracy" ${config.scoring === 'accuracy' ? 'selected' : ''}>准确率</option>
                    <option value="f1" ${config.scoring === 'f1' ? 'selected' : ''}>F1分数</option>
                    <option value="roc_auc" ${config.scoring === 'roc_auc' ? 'selected' : ''}>ROC AUC</option>
                    <option value="r2" ${config.scoring === 'r2' ? 'selected' : ''}>R2分数</option>
                </select>
            </div>
            <div class="form-group">
                <label>参数搜索空间 (JSON格式):</label>
                <textarea id="config-custom-param-grid" class="form-control" rows="5" 
                    placeholder='例如: {"n_estimators": [50, 100], "max_depth": [5, 10]}'>${config.custom_param_grid ? JSON.stringify(config.custom_param_grid, null, 2) : ''}</textarea>
                <small class="form-text text-muted">定义需要搜索的参数网格。每个参数的值必须是列表。</small>
            </div>
            <div class="form-group">
                <label>
                    <input type="checkbox" id="config-use-recommended" 
                           ${config.use_recommended !== false ? 'checked' : ''}>
                    使用系统推荐参数网格 (如果未提供搜索空间)
                </label>
            </div>
        `, () => {
            let customParamGrid = null;
            const customParamGridStr = document.getElementById('config-custom-param-grid').value.trim();
            if (customParamGridStr) {
                try {
                    customParamGrid = JSON.parse(customParamGridStr);
                } catch (e) {
                    alert('自定义参数网格JSON格式错误: ' + e.message);
                    return;
                }
            }
            
            this.canvas.updateNodeConfig(nodeId, {
                method: document.getElementById('config-tuning-method').value,
                cv: parseInt(document.getElementById('config-cv-folds').value),
                n_iter: parseInt(document.getElementById('config-n-iter').value),
                scoring: document.getElementById('config-scoring').value,
                use_recommended: document.getElementById('config-use-recommended').checked,
                custom_param_grid: customParamGrid,
                hyperparameters: customParamGrid // 将其作为 hyperparameters 传递，会被 tuneHyperparameters 使用
            });
            this.hideModal();
        });
    }
    
    /**
     * 显示异常值检测配置
     */
    showOutlierDetectionConfig(nodeId, config) {
        this.showModal('异常值检测配置', `
            <div class="form-group">
                <label>检测方法:</label>
                <select id="config-outlier-method" class="form-control">
                    <option value="iqr" ${config.method === 'iqr' ? 'selected' : ''}>IQR方法</option>
                    <option value="zscore" ${config.method === 'zscore' ? 'selected' : ''}>Z-Score方法</option>
                    <option value="isolation_forest" ${config.method === 'isolation_forest' ? 'selected' : ''}>孤立森林</option>
                </select>
            </div>
            <div class="form-group">
                <label>阈值:</label>
                <input type="number" id="config-outlier-threshold" class="form-control" 
                       value="${config.threshold || 1.5}" step="0.1" min="0.5" max="5">
                <small class="form-text text-muted">IQR方法默认1.5, Z-Score方法默认3.0</small>
            </div>
            <div class="form-group">
                <label>处理方式:</label>
                <select id="config-outlier-handle" class="form-control">
                    <option value="detect" ${config.handle_method === 'detect' ? 'selected' : ''}>仅检测</option>
                    <option value="clip" ${config.handle_method === 'clip' ? 'selected' : ''}>裁剪到边界</option>
                    <option value="remove" ${config.handle_method === 'remove' ? 'selected' : ''}>删除异常值</option>
                </select>
            </div>
            <div class="form-group">
                <label>应用列 (逗号分隔,空表示所有数值列):</label>
                <input type="text" id="config-outlier-columns" class="form-control" 
                       value="${(config.columns || []).join(',')}" placeholder="例如: age,income,price">
            </div>
        `, () => {
            const columns = document.getElementById('config-outlier-columns').value
                .split(',').map(c => c.trim()).filter(c => c);
            
            this.canvas.updateNodeConfig(nodeId, {
                method: document.getElementById('config-outlier-method').value,
                threshold: parseFloat(document.getElementById('config-outlier-threshold').value),
                handle_method: document.getElementById('config-outlier-handle').value,
                columns: columns.length > 0 ? columns : null
            });
            this.hideModal();
        });
    }
    
    /**
     * 显示特征选择配置
     */
    showFeatureSelectionConfig(nodeId, config) {
        this.showModal('特征选择配置', `
            <div class="form-group">
                <label>选择方法:</label>
                <select id="config-selection-method" class="form-control">
                    <option value="variance" ${config.method === 'variance' ? 'selected' : ''}>方差阈值</option>
                    <option value="correlation" ${config.method === 'correlation' ? 'selected' : ''}>相关系数</option>
                    <option value="importance" ${config.method === 'importance' ? 'selected' : ''}>特征重要性</option>
                    <option value="recursive" ${config.method === 'recursive' ? 'selected' : ''}>递归特征消除</option>
                </select>
            </div>
            <div class="form-group">
                <label>目标列 (特征重要性/RFE需要):</label>
                <input type="text" id="config-target-column" class="form-control" 
                       value="${config.target_column || ''}" placeholder="例如: target">
            </div>
            <div class="form-group">
                <label>保留特征数 (空表示使用阈值):</label>
                <input type="number" id="config-n-features" class="form-control" 
                       value="${config.n_features || ''}" min="1" placeholder="例如: 10">
            </div>
            <div class="form-group">
                <label>阈值:</label>
                <input type="number" id="config-selection-threshold" class="form-control" 
                       value="${config.threshold || 0.01}" step="0.01" min="0">
                <small class="form-text text-muted">方差阈值默认0.01, 相关系数默认0.9</small>
            </div>
        `, () => {
            const nFeatures = document.getElementById('config-n-features').value;
            
            this.canvas.updateNodeConfig(nodeId, {
                method: document.getElementById('config-selection-method').value,
                target_column: document.getElementById('config-target-column').value,
                n_features: nFeatures ? parseInt(nFeatures) : null,
                threshold: parseFloat(document.getElementById('config-selection-threshold').value)
            });
            this.hideModal();
        });
    }
    
    /**
     * 显示输出模块配置
     */
    showOutputModuleConfig(nodeId, nodeName, config) {
        // 模型结果和可视化节点不需要配置,直接标记为已配置
        if (nodeName === '模型结果' || nodeName === '可视化') {
            // 直接更新节点状态为已配置
            this.canvas.updateNodeConfig(nodeId, { auto_configured: true });
            alert(`${nodeName}节点无需配置，将在工作流执行后自动显示结果。\n\n请直接点击节点查看属性面板以查看结果。`);
            return;
        }
        
        let content = '';
        
        if (nodeName === '终止') {
            content = `<p>终止节点无需配置</p>`;
        }
        
        this.showModal(`${nodeName} 配置`, content, () => {
            let newConfig = {};
            
            this.canvas.updateNodeConfig(nodeId, newConfig);
            this.hideModal();
        });
    }
    
    /**
     * 显示模态对话框
     */
    showModal(title, content, onConfirm) {
        const existingModal = document.getElementById('workflow-config-modal');
        if (existingModal) {
            existingModal.remove();
        }
        
        const modal = document.createElement('div');
        modal.id = 'workflow-config-modal';
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close" onclick="workflowManager.hideModal()">&times;</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="workflowManager.hideModal()">取消</button>
                    <button class="btn btn-primary" id="modal-confirm-btn">确认</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        document.getElementById('modal-confirm-btn').addEventListener('click', onConfirm);
    }
    
    /**
     * 隐藏模态对话框
     */
    hideModal() {
        const modal = document.getElementById('workflow-config-modal');
        if (modal) {
            modal.remove();
        }
    }
    
    /**
     * 执行工作流
     */
    async executeWorkflow() {
        if (this.isExecuting) {
            alert('工作流正在执行中...');
            return;
        }
        
        const workflowData = this.canvas.getWorkflowData();
        
        if (workflowData.nodes.length === 0) {
            alert('工作流为空，请添加节点');
            return;
        }
        
        // 验证工作流
        const validation = this.validateWorkflow(workflowData);
        if (!validation.valid) {
            alert('工作流验证失败: ' + validation.error);
            return;
        }
        
        this.isExecuting = true;
        this.updateExecutionStatus('执行中...');
        
        // 初始化节点输出缓存
        this.nodeOutputs = new Map();
        const allCreatedDatasetIds = new Set();
        const sourceDatasetIds = new Set();

        // 识别源数据集ID，防止被误删
        workflowData.nodes.forEach(node => {
            if ((node.nodeName === '数据集选择' || node.nodeName === '数据上传') && node.nodeConfig && node.nodeConfig.dataset_id) {
                sourceDatasetIds.add(node.nodeConfig.dataset_id);
            }
        });
        
        try {
            // 按拓扑顺序执行节点
            const executionOrder = this.getExecutionOrder(workflowData);
            
            for (const nodeId of executionOrder) {
                const result = await this.executeNode(nodeId, workflowData);
                
                // 收集生成的数据集ID
                if (result) {
                    if (result.created_dataset_ids) {
                        result.created_dataset_ids.forEach(id => allCreatedDatasetIds.add(id));
                    }
                    if (result.new_dataset_id) {
                        allCreatedDatasetIds.add(result.new_dataset_id);
                    }
                }
            }
            
            this.updateExecutionStatus('执行成功');
            
            // 清理中间数据集
            await this.cleanupIntermediateDatasets(workflowData, this.nodeOutputs, allCreatedDatasetIds, sourceDatasetIds);
            
            alert('工作流执行成功！');
            
        } catch (error) {
            console.error('工作流执行失败:', error);
            this.updateExecutionStatus('执行失败');
            alert('工作流执行失败: ' + error.message);
        } finally {
            this.isExecuting = false;
        }
    }
    
    /**
     * 验证工作流
     */
    validateWorkflow(workflowData) {
        // 检查是否有未配置的节点
        for (const node of workflowData.nodes) {
            const nodeObj = this.canvas.nodes.get(node.id);
            if (nodeObj && nodeObj.nodeStatus === 'unconfigured') {
                return {
                    valid: false,
                    error: `节点 "${node.name}" 未配置`
                };
            }
        }
        
        // 检查是否有孤立节点
        const connectedNodes = new Set();
        workflowData.connections.forEach(conn => {
            connectedNodes.add(conn.from);
            connectedNodes.add(conn.to);
        });
        
        // 允许单个节点作为起始节点
        if (workflowData.nodes.length > 1) {
            for (const node of workflowData.nodes) {
                if (!connectedNodes.has(node.id)) {
                    return {
                        valid: false,
                        error: `节点 "${node.name}" 未连接到工作流`
                    };
                }
            }
        }
        
        return { valid: true };
    }
    
    /**
     * 获取执行顺序（拓扑排序）
     */
    getExecutionOrder(workflowData) {
        const graph = new Map();
        const inDegree = new Map();
        
        // 初始化图
        workflowData.nodes.forEach(node => {
            graph.set(node.id, []);
            inDegree.set(node.id, 0);
        });
        
        // 构建图
        workflowData.connections.forEach(conn => {
            graph.get(conn.from).push(conn.to);
            inDegree.set(conn.to, inDegree.get(conn.to) + 1);
        });
        
        // 拓扑排序
        const queue = [];
        const result = [];
        
        inDegree.forEach((degree, nodeId) => {
            if (degree === 0) {
                queue.push(nodeId);
            }
        });
        
        while (queue.length > 0) {
            const nodeId = queue.shift();
            result.push(nodeId);
            
            graph.get(nodeId).forEach(nextNodeId => {
                inDegree.set(nextNodeId, inDegree.get(nextNodeId) - 1);
                if (inDegree.get(nextNodeId) === 0) {
                    queue.push(nextNodeId);
                }
            });
        }
        
        return result;
    }
    
    /**
     * 执行单个节点
     */
    async executeNode(nodeId, workflowData) {
        const node = this.canvas.nodes.get(nodeId);
        if (!node) return;
        
        this.canvas.updateNodeStatus(nodeId, 'running', '运行中...');
        
        try {
            // 根据节点类型调用不同的API
            const nodeName = node.nodeName;
            // 深拷贝配置，避免修改原始配置
            let config = JSON.parse(JSON.stringify(node.nodeConfig || {}));
            
            // 获取上游节点输出并合并到当前配置
            if (workflowData && this.nodeOutputs) {
                const upstreamNodeIds = workflowData.connections
                    .filter(conn => conn.to === nodeId)
                    .map(conn => conn.from);
                
                if (upstreamNodeIds.length > 0) {
                    console.group(`节点 ${nodeId} (${nodeName}) 上游输入诊断`);
                    console.log('当前配置(config):', JSON.parse(JSON.stringify(config)));
                }

                for (const upId of upstreamNodeIds) {
                    const output = this.nodeOutputs.get(upId);
                    if (upstreamNodeIds.length > 0) {
                         console.log(`上游节点 ${upId} 输出:`, output);
                    }

                    if (output) {
                        // 传递 dataset_id
                        if (output.new_dataset_id) {
                            config.dataset_id = output.new_dataset_id;
                            console.log(`从上游 ${upId} 继承 new_dataset_id:`, output.new_dataset_id);
                        } else if (output.dataset_id) {
                            config.dataset_id = output.dataset_id;
                            console.log(`从上游 ${upId} 继承 dataset_id:`, output.dataset_id);
                        }
                        
                        // 传递 model_id
                        if (output.model_id) {
                            config.model_id = output.model_id;
                        } else if (output.id && (output.algorithm || output.algorithm_name || output.metrics)) {
                            // 兼容性处理：如果上游输出的是完整模型对象（如来自"模型结果"节点），使用其id作为model_id
                            config.model_id = output.id;
                            console.log(`从上游 ${upId} 推断 model_id:`, output.id);
                        }

                        // 传递 algorithm 信息
                        if (output.algorithm_name) {
                            config.upstream_algorithm_name = output.algorithm_name;
                        }
                        if (output.algorithm_type) {
                            config.upstream_algorithm_type = output.algorithm_type;
                        }
                        // 传递 hyperparameters 信息 (来自算法选择节点或超参数调优节点)
                        if (output.best_params) {
                            config.upstream_hyperparameters = output.best_params;
                        } else if (output.hyperparameters) {
                            config.upstream_hyperparameters = output.hyperparameters;
                        }
                        
                        // 传递 fixed_hyperparameters 信息 (来自算法选择节点)
                        if (output.fixed_hyperparameters) {
                            config.upstream_fixed_hyperparameters = output.fixed_hyperparameters;
                        }

                        // 传递 preprocessing_components
                        if (output.preprocessing_components) {
                            config.preprocessing_components = (config.preprocessing_components || []).concat(output.preprocessing_components);
                        }
                    }
                }

                if (upstreamNodeIds.length > 0) {
                    console.log('合并后配置:', config);
                    console.groupEnd();
                }
            }
            
            let result = null;
            
            switch(nodeName) {
                case '数据集选择':
                case '数据上传':
                    // 数据源节点，直接返回配置中的dataset_id
                    result = { dataset_id: config.dataset_id };
                    await new Promise(resolve => setTimeout(resolve, 500));
                    break;

                case '分类算法选择':
                case '回归算法选择':
                case '聚类算法选择':
                case '降维算法选择':
                    // 算法选择节点：不进行训练，仅传递配置
                    result = {
                        algorithm_name: config.algorithm,
                        algorithm_type: config.algorithm_type,
                        hyperparameters: config.hyperparameters,
                        fixed_hyperparameters: config.fixed_hyperparameters, // 传递固定参数
                        dataset_id: config.dataset_id
                    };
                    await new Promise(resolve => setTimeout(resolve, 500)); // 模拟短暂执行
                    break;

                case '模型训练':
                    // 模型训练节点：接收上游算法配置进行训练
                    if (!config.upstream_algorithm_name) {
                        throw new Error('模型训练节点必须连接到上游的算法选择节点或超参数调优节点');
                    }
                    
                    // 合并配置
                    const trainConfig = {
                        ...config,
                        algorithm: config.upstream_algorithm_name,
                        hyperparameters: config.upstream_hyperparameters || {}
                    };

                    // 确保上游的固定参数也能被合并到hyperparameters中
                    if (config.upstream_fixed_hyperparameters) {
                        trainConfig.hyperparameters = {
                            ...trainConfig.hyperparameters,
                            ...config.upstream_fixed_hyperparameters
                        };
                    }
                    
                    result = await this.trainModelAsync(trainConfig, nodeId);
                    break;

                case '分类算法':
                case '回归算法':
                case '聚类算法':
                case '降维算法':
                    // 异步训练模型（带进度监控）
                    result = await this.trainModelAsync(config, nodeId);
                    break;
                    
                case '超参数调优':
                    // 超参数调优
                    // 如果配置中没有指定算法，尝试使用上游传入的算法
                    if (!config.algorithm && config.upstream_algorithm_name) {
                        config.algorithm = config.upstream_algorithm_name;
                    }
                    // 优先使用上游传入的超参数/搜索空间
                    if (config.upstream_hyperparameters) {
                        config.hyperparameters = config.upstream_hyperparameters;
                    }
                    result = await this.tuneHyperparameters(config, nodeId);
                    break;
                    
                case '异常值检测':
                    // 异常值检测
                    result = await this.detectOutliers(config);
                    break;
                    
                case '特征选择':
                    // 特征选择
                    result = await this.selectFeatures(config);
                    break;
                    
                case '预处理':
                    // 数据预处理
                    result = await this.preprocessData(config);
                    break;

                case '模型结果':
                case '可视化':
                    // 获取模型结果
                    if (config.model_id) {
                        result = await this.getModelResults(config.model_id);
                        // 可以在这里触发UI更新，显示结果
                        this.showExecutionResult(nodeId, result);
                    } else {
                        console.warn('模型结果/可视化节点未接收到 model_id');
                        await new Promise(resolve => setTimeout(resolve, 500));
                        result = {};
                    }
                    break;
                    
                default:
                    // 其他节点模拟执行
                    await new Promise(resolve => setTimeout(resolve, 500));
                    result = {}; 
            }
            
            // 保存节点输出结果供下游使用
            if (result) {
                // 确保 dataset_id 被传递
                if (!result.dataset_id && config.dataset_id) {
                    result.dataset_id = config.dataset_id;
                }
                this.nodeOutputs.set(nodeId, result);
            }
            
            this.canvas.updateNodeStatus(nodeId, 'success', '成功');
            return result;
            
        } catch (error) {
            this.canvas.updateNodeStatus(nodeId, 'error', '失败');
            throw error;
        }
    }
    
    /**
     * 异步训练模型（带进度监控）
     */
    async trainModelAsync(config, nodeId) {
        const response = await fetch('/api/train/async', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: config.dataset_id,
                algorithm_name: config.algorithm,
                target_columns: config.target_columns,
                test_size: config.test_size,
                hyperparameters: config.hyperparameters,
                preprocessing_components: config.preprocessing_components
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '训练失败');
        }
        
        const taskId = result.task_id;
        
        // 监控进度
        return new Promise((resolve, reject) => {
            progressMonitor.startMonitoring(taskId, 'training', {
                onProgress: (progress) => {
                    this.canvas.updateNodeStatus(nodeId, 'running', progress.message);
                },
                onComplete: (progress) => {
                    resolve(progress.metadata.result);
                },
                onError: (error) => {
                    reject(new Error(error));
                }
            });
        });
    }
    
    /**
     * 超参数调优
     */
    async tuneHyperparameters(config, nodeId) {
        // 构造参数网格: 优先使用当前节点定义的custom_param_grid (存储在config.hyperparameters中)
        let finalParamGrid = config.hyperparameters || {};
        
        // 如果有上游传入的固定参数，合并到网格中
        // 注意：GridSearchCV要求所有参数值必须是列表
        if (config.upstream_fixed_hyperparameters) {
            // 深拷贝以避免修改原始对象
            finalParamGrid = JSON.parse(JSON.stringify(finalParamGrid));
            
            for (const [key, value] of Object.entries(config.upstream_fixed_hyperparameters)) {
                // 如果当前网格中没有该参数，则添加
                if (!(key in finalParamGrid)) {
                    // 如果值已经是列表，且看起来像搜索空间(多个值)，则保留
                    // 但这里是"固定参数"，所以应该被视为单个值
                    // 为了兼容GridSearchCV，必须包装成列表 [value]
                    finalParamGrid[key] = [value];
                }
            }
        }

        const response = await fetch('/api/hyperparameter/tune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: config.dataset_id,
                algorithm_name: config.algorithm,
                target_columns: config.target_columns,
                tuning_method: config.method || 'grid_search',
                cv: config.cv || 5,
                n_iter: config.n_iter || 10,
                scoring: config.scoring || 'accuracy',
                use_recommended: config.use_recommended !== false,
                param_grid: finalParamGrid
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '调优失败');
        }
        
        const taskId = result.task_id;
        
        // 监控进度
        return new Promise((resolve, reject) => {
            progressMonitor.startMonitoring(taskId, 'tuning', {
                onProgress: (progress) => {
                    this.canvas.updateNodeStatus(nodeId, 'running', progress.message);
                },
                onComplete: (progress) => {
                    resolve(progress.metadata.result);
                },
                onError: (error) => {
                    reject(new Error(error));
                }
            });
        });
    }
    
    /**
     * 异常值检测
     */
    async detectOutliers(config) {
        const response = await fetch('/api/preprocess/outliers/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: config.dataset_id,
                method: config.method || 'iqr',
                threshold: config.threshold || 1.5,
                columns: config.columns
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '异常值检测失败');
        }
        
        // 如果需要处理异常值
        if (config.handle_method && config.handle_method !== 'detect') {
            return this.handleOutliers(config, result.data.outliers);
        }
        
        return result.data;
    }
    
    /**
     * 处理异常值
     */
    async handleOutliers(config, outliers) {
        const response = await fetch('/api/preprocess/outliers/handle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: config.dataset_id,
                method: config.handle_method, // 'clip' 或 'remove'
                outliers: outliers
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '处理异常值失败');
        }
        
        return result.data;
    }
    
    /**
     * 特征选择
     */
    async selectFeatures(config) {
        const response = await fetch('/api/preprocess/features/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: config.dataset_id,
                method: config.method || 'variance',
                n_features: config.n_features || 10,
                threshold: config.threshold || 0.0,
                target_column: config.target_column
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '特征选择失败');
        }
        
        return result.data;
    }
    
    /**
     * 获取数据集信息
     */
    async getDatasetInfo(datasetId) {
        try {
            const response = await fetch(`/api/data/${datasetId}/info`);
            const result = await response.json();
            if (result.success) {
                return result.data;
            }
            return null;
        } catch (error) {
            console.error('获取数据集信息失败:', error);
            return null;
        }
    }
    
    /**
     * 数据预处理
     */
    async preprocessData(config) {
        let currentDatasetId = config.dataset_id;
        
        // ---------------------------------------------------------
        // [调试] 诊断 Dataset ID 为 1 的问题
        // ---------------------------------------------------------
        if (currentDatasetId == 1) {
            console.warn('⚠️ 检测到数据集ID为1，这可能是无效的默认值或配置丢失。');
            console.group('🔍 数据集ID诊断');
            console.log('当前传入配置(config):', JSON.parse(JSON.stringify(config)));
            console.log('config.dataset_id 来源:', config.dataset_id);
            
            // 尝试回溯
            if (config.dataset_id === 1) {
                console.trace('dataset_id 为 1 的调用栈');
            }

            console.log('当前工作流ID:', this.currentWorkflowId);
            
            // 尝试检查是否可以从上游恢复（仅供调试参考）
            if (this.nodeOutputs) {
                console.log('上游节点输出缓存:', Array.from(this.nodeOutputs.entries()));
            }
            console.groupEnd();
            
            // 提示用户检查
            if (window.UIHelper) {
                UIHelper.showMessage('检测到无效的数据集ID (1)。请检查"数据集选择"节点是否正确配置。', 'warning');
            }
        }
        // ---------------------------------------------------------

        const createdDatasetIds = [];
        const collectedComponents = [];
        let finalResult = {
            dataset_id: currentDatasetId
        };
        
        // 1. 自动编码 (根据数据管理页面的配置)
        // 优先执行自动编码，确保后续步骤处理的是数值型数据
        // 始终执行自动编码，以处理字符串列
        console.log(`开始自动编码，当前数据集ID: ${currentDatasetId}`);
        try {
            const response = await fetch('/api/preprocess/encode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: currentDatasetId,
                    workflow_id: this.currentWorkflowId,
                    target_column: config.target_column
                })
            });
            
            console.log('编码API响应状态:', response.status);
            
            // 如果响应不是200，直接抛出错误
            if (!response.ok) {
                let errorText = await response.text();
                console.error('编码API返回错误状态:', response.status, '响应:', errorText);
                throw new Error(`编码API返回错误状态: ${response.status}`);
            }
            
            let result;
            try {
                result = await response.json();
            } catch (jsonError) {
                console.error('解析编码API响应JSON失败:', jsonError);
                throw new Error(`编码API返回无效的JSON响应: ${jsonError.message}`);
            }
            console.log('编码API响应:', result);
            console.log('编码API响应成功:', result.success);
            
            if (result.success) {
                // 收集组件
                if (result.data.info && result.data.info.saved_encoders) {
                    console.log(`找到 ${result.data.info.saved_encoders.length} 个编码器`);
                    result.data.info.saved_encoders.forEach(enc => {
                        collectedComponents.push({
                            type: 'encoder',
                            name: `encoder_${enc.column}_${enc.method}`,
                            path: enc.path,
                            columns: [enc.column],
                            config: { method: enc.method }
                        });
                    });
                } else {
                    console.log('编码成功但没有保存编码器');
                }

                // 更新数据集ID
                if (result.data.new_dataset_id) {
                    console.log(`编码成功: 数据集ID从 ${currentDatasetId} 更新为 ${result.data.new_dataset_id}`);
                    currentDatasetId = result.data.new_dataset_id;
                    createdDatasetIds.push(currentDatasetId);
                    Object.assign(finalResult, result.data);
                } else {
                    console.error('编码成功但没有返回新的数据集ID，这可能导致后续步骤使用错误的数据集');
                    // 尝试从info中获取新数据集ID
                    if (result.data.info && result.data.info.new_dataset_id) {
                        console.log(`从info中找到新数据集ID: ${result.data.info.new_dataset_id}`);
                        currentDatasetId = result.data.info.new_dataset_id;
                        createdDatasetIds.push(currentDatasetId);
                    }
                }
            } else {
                console.error('编码失败:', result.error);
                // 如果编码失败，检查错误类型
                const errorMsg = result.error || '未知编码错误';
                
                // 先获取数据集信息，检查是否有字符串列
                const datasetInfo = await this.getDatasetInfo(currentDatasetId);
                if (datasetInfo) {
                    const hasStringColumns = datasetInfo.columns.some(col => 
                        col.data_type === 'categorical' || col.data_type === 'string'
                    );
                    
                    if (hasStringColumns) {
                        // 有字符串列但编码失败，必须抛出错误
                        throw new Error(`自动编码失败: ${errorMsg}. 数据集包含分类/字符串列，必须进行编码才能继续。`);
                    } else {
                        // 没有字符串列，编码不是必需的，但记录警告
                        console.warn('自动编码跳过: 数据集没有需要编码的分类列，错误:', errorMsg);
                    }
                } else {
                    // 无法获取数据集信息，保守起见抛出错误
                    throw new Error(`自动编码失败: ${errorMsg}. 无法获取数据集信息，无法确定是否需要编码。`);
                }
            }
        } catch (error) {
            console.error('自动编码请求失败:', error);
            // 编码是必需步骤，如果失败应该停止工作流
            // 先检查数据集是否有字符串列
            try {
                const datasetInfo = await this.getDatasetInfo(currentDatasetId);
                if (datasetInfo) {
                    const hasStringColumns = datasetInfo.columns.some(col => 
                        col.data_type === 'categorical' || col.data_type === 'string'
                    );
                    if (hasStringColumns) {
                        throw new Error(`自动编码请求失败: ${error.message}. 数据集包含分类/字符串列，编码是必需的。`);
                    } else {
                        console.warn('自动编码请求失败，但数据集没有字符串列，继续执行:', error.message);
                    }
                } else {
                    // 无法获取数据集信息，保守起见抛出错误
                    throw new Error(`自动编码请求失败: ${error.message}. 无法确定数据集是否需要编码。`);
                }
            } catch (infoError) {
                // 获取数据集信息也失败，抛出原始错误
                throw new Error(`自动编码请求失败: ${error.message}`);
            }
        }
        
        console.log(`编码后当前数据集ID: ${currentDatasetId}`);

        // 2. 处理缺失值
        if (config.handle_missing) {
            console.log(`开始处理缺失值，当前数据集ID: ${currentDatasetId}`);
            try {
                const response = await fetch('/api/preprocess/missing-values', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: currentDatasetId,
                        workflow_id: this.currentWorkflowId,
                        strategy: config.missing_strategy || 'mean'
                    })
                });
                
                console.log('缺失值处理API响应状态:', response.status);
                
                if (!response.ok) {
                    let errorText = await response.text();
                    console.error('缺失值处理API返回错误状态:', response.status, '响应:', errorText);
                    throw new Error(`缺失值处理API返回错误状态: ${response.status}`);
                }
                
                const result = await response.json();
                console.log('缺失值处理API响应:', result);
                
                if (!result.success) {
                    throw new Error(result.error || '处理缺失值失败');
                }
                
                // 收集组件
                if (result.data.info && result.data.info.saved_imputers) {
                    console.log(`找到 ${result.data.info.saved_imputers.length} 个填充器`);
                    result.data.info.saved_imputers.forEach(imp => {
                        collectedComponents.push({
                            type: 'imputer',
                            name: `imputer_${imp.strategy}`,
                            path: imp.path,
                            columns: imp.columns,
                            config: { strategy: imp.strategy }
                        });
                    });
                }
                
                // 更新数据集ID，供下一步骤使用
                if (result.data.new_dataset_id) {
                    console.log(`缺失值处理成功: 数据集ID从 ${currentDatasetId} 更新为 ${result.data.new_dataset_id}`);
                    currentDatasetId = result.data.new_dataset_id;
                    createdDatasetIds.push(currentDatasetId);
                    Object.assign(finalResult, result.data);
                } else {
                    console.error('缺失值处理成功但没有返回新的数据集ID，这可能导致后续步骤使用错误的数据集');
                    // 尝试从info中获取新数据集ID
                    if (result.data.info && result.data.info.new_dataset_id) {
                        console.log(`从info中找到新数据集ID: ${result.data.info.new_dataset_id}`);
                        currentDatasetId = result.data.info.new_dataset_id;
                        createdDatasetIds.push(currentDatasetId);
                    }
                }
            } catch (error) {
                console.error('处理缺失值步骤失败:', error);
                throw error;
            }
            console.log(`缺失值处理后当前数据集ID: ${currentDatasetId}`);
        }
        
        // 3. 数据缩放
        if (config.scale) {
            console.log(`开始数据缩放，当前数据集ID: ${currentDatasetId}`);
            const response = await fetch('/api/preprocess/scale', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: currentDatasetId,
                    workflow_id: this.currentWorkflowId,
                    method: config.scale_method || 'standard'
                })
            });
            
            console.log('数据缩放API响应状态:', response.status);
            
            if (!response.ok) {
                let errorText = await response.text();
                console.error('数据缩放API返回错误状态:', response.status, '响应:', errorText);
                throw new Error(`数据缩放API返回错误状态: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('数据缩放API响应:', result);
            
            if (!result.success) {
                throw new Error(result.error || '数据缩放失败');
            }
            
            // 收集组件
            if (result.data.info && result.data.info.saved_scaler) {
                const scaler = result.data.info.saved_scaler;
                console.log(`找到缩放器: ${scaler.method}`);
                collectedComponents.push({
                    type: 'scaler',
                    name: `scaler_${scaler.method}`,
                    path: scaler.path,
                    columns: scaler.columns,
                    config: { method: scaler.method }
                });
            }
            
            // 更新数据集ID
            if (result.data.new_dataset_id) {
                console.log(`数据缩放成功: 数据集ID从 ${currentDatasetId} 更新为 ${result.data.new_dataset_id}`);
                currentDatasetId = result.data.new_dataset_id;
                createdDatasetIds.push(currentDatasetId);
                Object.assign(finalResult, result.data);
            } else {
                console.error('数据缩放成功但没有返回新的数据集ID');
            }
            console.log(`数据缩放后当前数据集ID: ${currentDatasetId}`);
        }
        
        // 确保最终结果包含最新的 dataset_id 和收集到的组件
        finalResult.new_dataset_id = currentDatasetId;
        finalResult.dataset_id = currentDatasetId;
        finalResult.created_dataset_ids = createdDatasetIds;
        finalResult.preprocessing_components = collectedComponents;
        
        return finalResult;
    }
    
    /**
     * 获取模型结果
     */
    async getModelResults(modelId) {
        // 获取完整模型详情，包含算法信息和完整结果
        const response = await fetch(`/api/models/${modelId}`);
        const json = await response.json();
        if (json.success) {
            return json.data;
        }
        throw new Error(json.error || '获取模型结果失败');
    }

    /**
     * 显示执行结果
     */
    showExecutionResult(nodeId, result) {
        console.log('节点执行结果:', result);
        
        const node = this.canvas.nodes.get(nodeId);
        if (node && result) {
            // 将结果存储到节点数据中
            node.executionResult = result;
            
            // 更新节点状态显示
            const items = node.getObjects();
            if (items.length >= 4) {
                items[3].set('text', '点击查看结果');
                items[3].set('fill', '#4CAF50');
                items[3].set('fontWeight', 'bold');
            }
            
            this.canvas.canvas.renderAll();
            
            // 如果该节点当前被选中，刷新属性面板
            if (this.canvas.selectedNode === node) {
                this.canvas.showNodeProperties(nodeId);
            }
        }
    }

    /**
     * 更新执行状态
     */
    updateExecutionStatus(status) {
        const statusBar = document.querySelector('.status-bar .status-left span');
        if (statusBar) {
            statusBar.textContent = status;
        }
    }
    
    /**
     * 保存工作流
     */
    async saveWorkflow() {
        const workflowData = this.canvas.getWorkflowData();
        
        if (workflowData.nodes.length === 0) {
            alert('工作流为空');
            return;
        }
        
        let name = prompt('请输入工作流名称:', this.currentWorkflowName || '我的工作流');
        if (!name) return;
        
        const description = prompt('请输入工作流描述 (可选):', this.currentWorkflowDescription || '');
        
        try {
            // 确定工作流类型 (根据最后一个节点推断，或默认为 general)
            let type = 'general';
            // 简单逻辑: 查找是否有算法节点
            const hasAlgo = workflowData.nodes.some(n => n.type && n.type.includes('算法'));
            if (hasAlgo) type = 'training';
            
            const payload = {
                name: name,
                workflow_type: type,
                description: description,
                configuration: workflowData
            };
            
            let result;
            if (this.currentWorkflowId) {
                // 更新
                result = await apiClient.updateWorkflow(this.currentWorkflowId, payload);
            } else {
                // 创建
                result = await apiClient.createWorkflow(payload);
                if (result.success) {
                    this.currentWorkflowId = result.data.workflow_id;
                }
            }
            
            if (result.success) {
                this.currentWorkflowName = name;
                this.currentWorkflowDescription = description;
                alert('工作流保存成功！');
            } else {
                throw new Error(result.error || '保存失败');
            }
            
        } catch (error) {
            console.error('保存工作流失败:', error);
            alert('保存工作流失败: ' + error.message);
        }
    }

    /**
     * 显示工作流管理对话框
     */
    showWorkflowManager() {
        const modal = document.getElementById('workflow-manage-modal');
        if (modal) {
            modal.classList.add('show');
            this.loadWorkflows();
        }
    }

    /**
     * 隐藏工作流管理对话框
     */
    hideWorkflowManager() {
        const modal = document.getElementById('workflow-manage-modal');
        if (modal) {
            modal.classList.remove('show');
        }
    }

    /**
     * 加载工作流列表
     */
    async loadWorkflows() {
        const listEl = document.getElementById('workflow-list');
        listEl.innerHTML = '<div class="loading-spinner">加载中...</div>';
        
        try {
            const result = await apiClient.getWorkflows();
            
            if (result.success && result.data.length > 0) {
                listEl.innerHTML = '';
                result.data.forEach(workflow => {
                    const item = this.createWorkflowItem(workflow);
                    listEl.appendChild(item);
                });
                this.updateBatchDeleteWorkflowButton();
            } else {
                listEl.innerHTML = '<div class="empty-hint"><p>暂无工作流</p></div>';
            }
        } catch (error) {
            console.error('加载工作流列表失败:', error);
            listEl.innerHTML = '<div class="empty-hint"><p>加载失败</p></div>';
        }
    }

    /**
     * 创建工作流列表项
     */
    createWorkflowItem(workflow) {
        const item = document.createElement('div');
        item.className = 'workflow-item'; // 需要在CSS中定义样式，或复用 dataset-item 样式
        // 复用 dataset-item 样式
        item.classList.add('dataset-item');
        item.dataset.id = workflow.id;
        
        // Checkbox
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'workflow-checkbox';
        checkbox.value = workflow.id;
        checkbox.style.marginRight = '10px';
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
            this.updateBatchDeleteWorkflowButton();
        });
        
        item.innerHTML = `
            <div class="dataset-item-name" style="flex:1">
                <strong>${workflow.name}</strong>
                <div style="font-size:0.8em; color:#666;">${workflow.description || '无描述'}</div>
            </div>
            <div class="dataset-item-info">
                ${new Date(workflow.created_at).toLocaleDateString()}
            </div>
            <div class="item-actions" style="margin-left: 10px;">
                <button class="btn btn-sm btn-primary btn-load-workflow" style="margin-right:5px;">加载</button>
                <button class="btn btn-sm btn-danger btn-delete-workflow">删除</button>
            </div>
        `;
        
        item.prepend(checkbox);
        
        // 绑定按钮事件
        const loadBtn = item.querySelector('.btn-load-workflow');
        loadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.loadWorkflow(workflow.id);
        });
        
        const deleteBtn = item.querySelector('.btn-delete-workflow');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteWorkflow(workflow.id, workflow.name);
        });
        
        return item;
    }

    /**
     * 更新批量删除按钮状态
     */
    updateBatchDeleteWorkflowButton() {
        const checkboxes = document.querySelectorAll('.workflow-checkbox:checked');
        const btn = document.getElementById('btn-batch-delete-workflows');
        if (btn) {
            btn.style.display = checkboxes.length > 0 ? 'inline-block' : 'none';
        }
    }

    /**
     * 加载单个工作流
     */
    async loadWorkflow(id) {
        try {
            const result = await apiClient.getWorkflow(id);
            if (result.success) {
                const workflow = result.data;
                this.currentWorkflowId = workflow.id;
                this.currentWorkflowName = workflow.name;
                this.currentWorkflowDescription = workflow.description;
                
                // 清空当前画布
                this.canvas.clear();
                
                // 加载配置
                if (workflow.configuration) {
                    this.canvas.loadWorkflowData(workflow.configuration);
                }
                
                this.hideWorkflowManager();
                alert(`工作流 "${workflow.name}" 加载成功`);
            } else {
                throw new Error(result.error || '加载失败');
            }
        } catch (error) {
            console.error('加载工作流失败:', error);
            alert('加载工作流失败: ' + error.message);
        }
    }

    /**
     * 删除工作流
     */
    async deleteWorkflow(id, name) {
        if (!confirm(`确定要删除工作流 "${name}" 吗？`)) {
            return;
        }
        
        try {
            const result = await apiClient.deleteWorkflow(id);
            if (result.success) {
                // 如果当前正在编辑该工作流，清除ID
                if (this.currentWorkflowId == id) {
                    this.currentWorkflowId = null;
                    this.currentWorkflowName = null;
                    this.currentWorkflowDescription = null;
                }
                this.loadWorkflows();
            } else {
                throw new Error(result.error || '删除失败');
            }
        } catch (error) {
            console.error('删除工作流失败:', error);
            alert('删除工作流失败: ' + error.message);
        }
    }

    /**
     * 批量删除工作流
     */
    async batchDeleteWorkflows() {
        const checkboxes = document.querySelectorAll('.workflow-checkbox:checked');
        if (checkboxes.length === 0) return;

        if (!confirm(`确定要删除选中的 ${checkboxes.length} 个工作流吗？`)) {
            return;
        }

        const ids = Array.from(checkboxes).map(cb => parseInt(cb.value));

        try {
            const result = await apiClient.batchDeleteWorkflows(ids);
            if (result.success) {
                // 如果包含当前工作流
                if (this.currentWorkflowId && ids.includes(this.currentWorkflowId)) {
                    this.currentWorkflowId = null;
                    this.currentWorkflowName = null;
                    this.currentWorkflowDescription = null;
                }
                this.loadWorkflows();
                alert('批量删除成功');
            } else {
                throw new Error(result.error || '删除失败');
            }
        } catch (error) {
            console.error('批量删除失败:', error);
            alert('批量删除失败: ' + error.message);
        }
    }

    /**
     * 清理中间数据集
     */
    async cleanupIntermediateDatasets(workflowData, nodeOutputs, allCreatedDatasetIds, sourceDatasetIds) {
        if (allCreatedDatasetIds.size === 0) return;

        console.log('开始清理中间数据集...');
        const idsToKeep = new Set();

        // 找出所有有出边的节点
        const sourceNodes = new Set();
        if (workflowData.connections) {
            workflowData.connections.forEach(conn => {
                sourceNodes.add(conn.from);
            });
        }

        // 遍历所有节点
        workflowData.nodes.forEach(node => {
            // 如果该节点没有作为源节点（即没有出边），则它是终点节点，其输出应保留
            if (!sourceNodes.has(node.id)) {
                const output = nodeOutputs.get(node.id);
                let datasetId = null;
                
                if (output) {
                    // 尝试从不同结构中获取 dataset_id
                    if (typeof output === 'object') {
                        datasetId = output.new_dataset_id || output.dataset_id;
                    } else {
                        // 可能是直接的 ID 值
                        datasetId = output;
                    }
                }
                
                if (datasetId) {
                    idsToKeep.add(datasetId);
                }
            }
        });
        
        const idsToDelete = [...allCreatedDatasetIds].filter(id => {
            // 如果是需要保留的数据集，不删除
            if (idsToKeep.has(id)) return false;
            // 如果是源数据集，绝对不删除
            if (sourceDatasetIds && sourceDatasetIds.has(id)) {
                console.log(`保留源数据集: ${id}`);
                return false;
            }
            return true;
        });
        
        if (idsToDelete.length > 0) {
            console.log('正在删除中间数据集:', idsToDelete);
            await this.batchDeleteDatasets(idsToDelete);
        } else {
            console.log('没有中间数据集需要清理');
        }
    }

    /**
     * 批量删除数据集
     */
    async batchDeleteDatasets(datasetIds) {
        try {
            const response = await fetch('/api/data/datasets/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_ids: datasetIds })
            });
            
            const result = await response.json();
            if (!result.success) {
                console.warn('批量删除部分失败:', result.error);
            }
        } catch (error) {
            console.error('批量删除请求失败:', error);
        }
    }
}

// 导出
window.WorkflowManager = WorkflowManager;
