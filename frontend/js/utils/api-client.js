/**
 * API客户端工具
 * 统一管理所有API请求
 */

class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    /**
     * 发送GET请求
     */
    async get(url, params = {}) {
        // 添加时间戳防止缓存
        const paramsWithCacheBuster = { ...params, _t: Date.now() };
        const queryString = new URLSearchParams(paramsWithCacheBuster).toString();
        const fullURL = queryString ? `${this.baseURL}${url}?${queryString}` : `${this.baseURL}${url}`;
        
        try {
            const response = await fetch(fullURL, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('GET请求失败:', error);
            throw error;
        }
    }

    /**
     * 发送POST请求
     */
    async post(url, data = {}) {
        try {
            const response = await fetch(`${this.baseURL}${url}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('POST请求失败:', error);
            throw error;
        }
    }

    /**
     * 发送PUT请求
     */
    async put(url, data = {}) {
        try {
            const response = await fetch(`${this.baseURL}${url}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('PUT请求失败:', error);
            throw error;
        }
    }

    /**
     * 发送DELETE请求
     */
    async delete(url) {
        try {
            const response = await fetch(`${this.baseURL}${url}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('DELETE请求失败:', error);
            throw error;
        }
    }

    /**
     * 上传文件
     */
    async upload(url, formData) {
        try {
            const response = await fetch(`${this.baseURL}${url}`, {
                method: 'POST',
                body: formData
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('文件上传失败:', error);
            throw error;
        }
    }

    /**
     * 处理响应
     */
    async handleResponse(response) {
        const data = await response.json();
        
        if (response.ok) {
            return data;
        } else {
            throw new Error(data.error || `请求失败: ${response.status}`);
        }
    }

    // ==================== 数据管理API ====================
    
    async uploadDataset(formData) {
        return this.upload('/api/data/upload', formData);
    }

    async getDatasets() {
        return this.get('/api/data/datasets', { _t: Date.now() });
    }

    async getDataset(id) {
        return this.get(`/api/data/datasets/${id}`);
    }

    async getDatasetData(id, page = 1, pageSize = 100) {
        return this.get(`/api/data/datasets/${id}/data`, { 
            page, 
            page_size: pageSize,
            _t: Date.now() // Prevent caching
        });
    }

    async saveDatasetData(id, data) {
        return this.post(`/api/data/datasets/${id}/data`, data);
    }

    async updateColumn(datasetId, columnId, data) {
        return this.put(`/api/data/columns/${columnId}`, data);
    }

    async deleteDataset(id) {
        return this.delete(`/api/data/datasets/${id}`);
    }

    async batchDeleteDatasets(datasetIds) {
        return this.post('/api/data/datasets/batch', { dataset_ids: datasetIds });
    }

    // ==================== 模型管理API ====================
    
    async getModels() {
        return this.get('/api/models');
    }

    async getModel(id) {
        return this.get(`/api/models/${id}`);
    }

    async getModelResults(id) {
        return this.get(`/api/models/${id}/results`);
    }

    async deleteModel(id) {
        return this.delete(`/api/models/${id}`);
    }

    async getAlgorithms() {
        return this.get('/api/algorithms');
    }

    async getAlgorithmInfo(name) {
        return this.get(`/api/algorithms/${name}`);
    }

    async getAlgorithmHyperparameters(name) {
        return this.get(`/api/algorithms/${name}/hyperparameters`);
    }

    // ==================== 训练API ====================
    
    async trainModel(data) {
        return this.post('/api/train', data);
    }

    // ==================== 预测API ====================
    
    async predict(data) {
        return this.post('/api/predict', data);
    }

    async batchPredict(data) {
        return this.post('/api/predict/batch', data);
    }

    async getPredictions(params = {}) {
        return this.get('/api/predictions', params);
    }

    async getPrediction(id) {
        return this.get(`/api/predictions/${id}`);
    }

    // ==================== 预处理API ====================
    
    async handleMissingValues(data) {
        return this.post('/api/preprocess/missing-values', data);
    }

    async scaleFeatures(data) {
        return this.post('/api/preprocess/scale', data);
    }

    async detectOutliers(data) {
        return this.post('/api/preprocess/outliers/detect', data);
    }

    async handleOutliers(data) {
        return this.post('/api/preprocess/outliers/handle', data);
    }

    async selectFeatures(data) {
        return this.post('/api/preprocess/features/select', data);
    }

    // ==================== 工作流API ====================
    
    async getWorkflows() {
        return this.get('/api/workflows');
    }

    async getWorkflow(id) {
        return this.get(`/api/workflows/${id}`);
    }

    async createWorkflow(data) {
        return this.post('/api/workflows', data);
    }

    async updateWorkflow(id, data) {
        return this.put(`/api/workflows/${id}`, data);
    }

    async deleteWorkflow(id) {
        return this.delete(`/api/workflows/${id}`);
    }

    async batchDeleteWorkflows(workflowIds) {
        return this.post('/api/workflows/batch', { workflow_ids: workflowIds });
    }

    // ==================== 健康检查 ====================
    
    async healthCheck() {
        return this.get('/api/health');
    }

    async getHealth() {
        return this.get('/api/health');
    }
}

// 创建全局API客户端实例
const apiClient = new APIClient();
