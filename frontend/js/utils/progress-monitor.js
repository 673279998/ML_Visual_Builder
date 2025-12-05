/**
 * 进度监控器 - 用于监控长时间运行的任务
 */

class ProgressMonitor {
    constructor() {
        this.activeMonitors = new Map();
        this.pollingInterval = 1000; // 轮询间隔：1秒
    }
    
    /**
     * 启动任务监控
     * @param {string} taskId - 任务ID
     * @param {string} taskType - 任务类型 (training, tuning, prediction等)
     * @param {Object} options - 配置选项
     * @param {Function} options.onProgress - 进度更新回调
     * @param {Function} options.onComplete - 完成回调
     * @param {Function} options.onError - 错误回调
     * @param {Function} options.onCancel - 取消回调
     * @param {number} options.interval - 自定义轮询间隔(ms)
     */
    startMonitoring(taskId, taskType, options = {}) {
        if (this.activeMonitors.has(taskId)) {
            console.warn(`任务 ${taskId} 已在监控中`);
            return;
        }
        
        const monitor = {
            taskId,
            taskType,
            options,
            interval: options.interval || this.pollingInterval,
            timerId: null,
            lastProgress: null
        };
        
        this.activeMonitors.set(taskId, monitor);
        
        // 立即执行一次检查
        this._checkProgress(taskId);
        
        // 启动定时轮询
        monitor.timerId = setInterval(() => {
            this._checkProgress(taskId);
        }, monitor.interval);
        
        console.log(`开始监控任务: ${taskId} (${taskType})`);
    }
    
    /**
     * 停止任务监控
     * @param {string} taskId - 任务ID
     */
    stopMonitoring(taskId) {
        const monitor = this.activeMonitors.get(taskId);
        if (!monitor) return;
        
        if (monitor.timerId) {
            clearInterval(monitor.timerId);
        }
        
        this.activeMonitors.delete(taskId);
        console.log(`停止监控任务: ${taskId}`);
    }
    
    /**
     * 取消任务
     * @param {string} taskId - 任务ID
     */
    async cancelTask(taskId) {
        const monitor = this.activeMonitors.get(taskId);
        if (!monitor) {
            throw new Error('任务不存在或未在监控中');
        }
        
        const endpoint = this._getCancelEndpoint(monitor.taskType, taskId);
        
        try {
            const response = await fetch(endpoint, { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.stopMonitoring(taskId);
                if (monitor.options.onCancel) {
                    monitor.options.onCancel(taskId);
                }
                return true;
            } else {
                throw new Error(result.error || '取消任务失败');
            }
        } catch (error) {
            console.error(`取消任务失败: ${error.message}`);
            throw error;
        }
    }
    
    /**
     * 获取所有活动监控
     */
    getActiveMonitors() {
        return Array.from(this.activeMonitors.values()).map(m => ({
            taskId: m.taskId,
            taskType: m.taskType,
            lastProgress: m.lastProgress
        }));
    }
    
    /**
     * 检查任务进度（内部方法）
     * @private
     */
    async _checkProgress(taskId) {
        const monitor = this.activeMonitors.get(taskId);
        if (!monitor) return;
        
        const endpoint = this._getProgressEndpoint(monitor.taskType, taskId);
        
        try {
            const response = await fetch(endpoint);
            const result = await response.json();
            
            if (!result.success) {
                this._handleError(taskId, result.error || '获取进度失败');
                return;
            }
            
            const progress = result.data;
            monitor.lastProgress = progress;
            
            // 触发进度更新回调
            if (monitor.options.onProgress) {
                monitor.options.onProgress(progress);
            }
            
            // 检查任务状态
            switch (progress.status) {
                case 'completed':
                    this._handleComplete(taskId, progress);
                    break;
                case 'failed':
                    this._handleError(taskId, progress.error || '任务失败');
                    break;
                case 'cancelled':
                    this._handleCancel(taskId, progress);
                    break;
                case 'running':
                    // 继续监控
                    break;
            }
            
        } catch (error) {
            console.error(`检查进度失败: ${error.message}`);
            // 不停止监控，继续尝试
        }
    }
    
    /**
     * 处理任务完成
     * @private
     */
    _handleComplete(taskId, progress) {
        const monitor = this.activeMonitors.get(taskId);
        if (!monitor) return;
        
        this.stopMonitoring(taskId);
        
        if (monitor.options.onComplete) {
            monitor.options.onComplete(progress);
        }
    }
    
    /**
     * 处理任务错误
     * @private
     */
    _handleError(taskId, error) {
        const monitor = this.activeMonitors.get(taskId);
        if (!monitor) return;
        
        this.stopMonitoring(taskId);
        
        if (monitor.options.onError) {
            monitor.options.onError(error);
        }
    }
    
    /**
     * 处理任务取消
     * @private
     */
    _handleCancel(taskId, progress) {
        const monitor = this.activeMonitors.get(taskId);
        if (!monitor) return;
        
        this.stopMonitoring(taskId);
        
        if (monitor.options.onCancel) {
            monitor.options.onCancel(progress);
        }
    }
    
    /**
     * 获取进度查询端点
     * @private
     */
    _getProgressEndpoint(taskType, taskId) {
        const endpoints = {
            'training': `/api/train/progress/${taskId}`,
            'tuning': `/api/hyperparameter/tune/progress/${taskId}`,
            'prediction': `/api/predict/progress/${taskId}`
        };
        
        return endpoints[taskType] || `/api/progress/${taskId}`;
    }
    
    /**
     * 获取取消任务端点
     * @private
     */
    _getCancelEndpoint(taskType, taskId) {
        const endpoints = {
            'training': `/api/train/cancel/${taskId}`,
            'tuning': `/api/hyperparameter/tune/cancel/${taskId}`,
            'prediction': `/api/predict/cancel/${taskId}`
        };
        
        return endpoints[taskType] || `/api/cancel/${taskId}`;
    }
}

/**
 * 创建进度条UI组件
 */
class ProgressBar {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`容器元素 ${containerId} 不存在`);
        }
        
        this.render();
    }
    
    /**
     * 渲染进度条
     */
    render() {
        this.container.innerHTML = `
            <div class="progress-container">
                <div class="progress-header">
                    <span class="progress-title">任务进度</span>
                    <button class="progress-cancel-btn" style="display: none;">取消</button>
                </div>
                <div class="progress-bar-wrapper">
                    <div class="progress-bar" style="width: 0%"></div>
                    <span class="progress-percentage">0%</span>
                </div>
                <div class="progress-message">准备中...</div>
                <div class="progress-details" style="display: none;">
                    <div class="progress-step">步骤: <span class="step-info">0/0</span></div>
                    <div class="progress-time">用时: <span class="time-info">0s</span></div>
                </div>
            </div>
        `;
        
        this.elements = {
            title: this.container.querySelector('.progress-title'),
            cancelBtn: this.container.querySelector('.progress-cancel-btn'),
            bar: this.container.querySelector('.progress-bar'),
            percentage: this.container.querySelector('.progress-percentage'),
            message: this.container.querySelector('.progress-message'),
            details: this.container.querySelector('.progress-details'),
            stepInfo: this.container.querySelector('.step-info'),
            timeInfo: this.container.querySelector('.time-info')
        };
    }
    
    /**
     * 更新进度
     * @param {Object} progress - 进度信息
     */
    update(progress) {
        const { status, progress: percent, message, current_step, total_steps, start_time } = progress;
        
        // 更新进度条
        this.elements.bar.style.width = `${percent}%`;
        this.elements.percentage.textContent = `${percent}%`;
        
        // 更新消息
        this.elements.message.textContent = message || '处理中...';
        
        // 更新步骤信息
        if (current_step !== undefined && total_steps !== undefined) {
            this.elements.stepInfo.textContent = `${current_step}/${total_steps}`;
            this.elements.details.style.display = 'block';
        }
        
        // 更新用时
        if (start_time) {
            const elapsed = Math.floor((new Date() - new Date(start_time)) / 1000);
            this.elements.timeInfo.textContent = `${elapsed}s`;
        }
        
        // 更新状态样式
        this.elements.bar.className = 'progress-bar';
        if (status === 'completed') {
            this.elements.bar.classList.add('success');
        } else if (status === 'failed') {
            this.elements.bar.classList.add('error');
        } else if (status === 'cancelled') {
            this.elements.bar.classList.add('cancelled');
        }
    }
    
    /**
     * 设置标题
     */
    setTitle(title) {
        this.elements.title.textContent = title;
    }
    
    /**
     * 设置取消按钮回调
     */
    onCancel(callback) {
        this.elements.cancelBtn.style.display = 'inline-block';
        this.elements.cancelBtn.onclick = callback;
    }
    
    /**
     * 显示进度条
     */
    show() {
        this.container.style.display = 'block';
    }
    
    /**
     * 隐藏进度条
     */
    hide() {
        this.container.style.display = 'none';
    }
    
    /**
     * 重置进度条
     */
    reset() {
        this.elements.bar.style.width = '0%';
        this.elements.percentage.textContent = '0%';
        this.elements.message.textContent = '准备中...';
        this.elements.details.style.display = 'none';
        this.elements.cancelBtn.style.display = 'none';
        this.elements.bar.className = 'progress-bar';
    }
}

// 全局进度监控实例
const progressMonitor = new ProgressMonitor();
