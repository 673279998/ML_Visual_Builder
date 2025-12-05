/**
 * UI辅助工具
 * 提供通用的UI组件和工具函数
 */

const UIHelper = {
    /**
     * 显示消息提示
     * @param {string} message - 消息内容
     * @param {string} type - 消息类型: success, error, warning, info
     * @param {number} duration - 显示时长(毫秒)
     */
    showMessage(message, type = 'info', duration = 3000) {
        // 创建消息容器（如果不存在）
        let container = document.getElementById('message-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'message-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            document.body.appendChild(container);
        }

        // 创建消息元素
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${type}`;
        messageEl.style.cssText = `
            padding: 12px 20px;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            color: white;
            font-size: 14px;
            min-width: 300px;
            animation: slideIn 0.3s ease-out;
        `;

        // 设置背景颜色
        const colors = {
            success: '#52c41a',
            error: '#f5222d',
            warning: '#faad14',
            info: '#1890ff'
        };
        messageEl.style.backgroundColor = colors[type] || colors.info;

        // 添加图标
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        messageEl.innerHTML = `<strong>${icons[type] || icons.info}</strong> ${message}`;

        // 添加到容器
        container.appendChild(messageEl);

        // 自动移除
        setTimeout(() => {
            messageEl.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                messageEl.remove();
                // 如果容器为空，移除容器
                if (container.children.length === 0) {
                    container.remove();
                }
            }, 300);
        }, duration);
    },

    /**
     * 显示加载动画
     * @param {string} message - 加载消息
     * @returns {Function} 关闭加载动画的函数
     */
    showLoading(message = '加载中...') {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        `;

        const loader = document.createElement('div');
        loader.style.cssText = `
            background: white;
            padding: 30px 40px;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        `;

        const spinner = document.createElement('div');
        spinner.style.cssText = `
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1890ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        `;

        const text = document.createElement('div');
        text.textContent = message;
        text.style.cssText = `
            color: #333;
            font-size: 14px;
        `;

        loader.appendChild(spinner);
        loader.appendChild(text);
        overlay.appendChild(loader);
        document.body.appendChild(overlay);

        // 添加旋转动画
        if (!document.getElementById('loading-styles')) {
            const style = document.createElement('style');
            style.id = 'loading-styles';
            style.textContent = `
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                @keyframes slideIn {
                    from {
                        opacity: 0;
                        transform: translateX(100%);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }
                @keyframes slideOut {
                    from {
                        opacity: 1;
                        transform: translateX(0);
                    }
                    to {
                        opacity: 0;
                        transform: translateX(100%);
                    }
                }
            `;
            document.head.appendChild(style);
        }

        // 返回关闭函数
        return () => {
            overlay.remove();
        };
    },

    /**
     * 确认对话框
     * @param {string} message - 确认消息
     * @param {string} title - 标题
     * @returns {Promise<boolean>} 用户选择
     */
    confirm(message, title = '确认') {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
            `;

            const dialog = document.createElement('div');
            dialog.style.cssText = `
                background: white;
                padding: 0;
                border-radius: 8px;
                min-width: 400px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            `;

            const header = document.createElement('div');
            header.style.cssText = `
                padding: 20px;
                border-bottom: 1px solid #e8e8e8;
                font-size: 16px;
                font-weight: 600;
            `;
            header.textContent = title;

            const body = document.createElement('div');
            body.style.cssText = `
                padding: 20px;
                font-size: 14px;
                color: #666;
            `;
            body.textContent = message;

            const footer = document.createElement('div');
            footer.style.cssText = `
                padding: 12px 20px;
                border-top: 1px solid #e8e8e8;
                display: flex;
                justify-content: flex-end;
                gap: 10px;
            `;

            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = '取消';
            cancelBtn.style.cssText = `
                padding: 8px 16px;
                border: 1px solid #d9d9d9;
                background: white;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            `;
            cancelBtn.onclick = () => {
                overlay.remove();
                resolve(false);
            };

            const confirmBtn = document.createElement('button');
            confirmBtn.textContent = '确定';
            confirmBtn.style.cssText = `
                padding: 8px 16px;
                border: none;
                background: #1890ff;
                color: white;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            `;
            confirmBtn.onclick = () => {
                overlay.remove();
                resolve(true);
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(confirmBtn);
            dialog.appendChild(header);
            dialog.appendChild(body);
            dialog.appendChild(footer);
            overlay.appendChild(dialog);
            document.body.appendChild(overlay);
        });
    },

    /**
     * 格式化日期
     * @param {string|Date} date - 日期
     * @returns {string} 格式化后的日期字符串
     */
    formatDate(date) {
        const d = new Date(date);
        return d.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    /**
     * 格式化文件大小
     * @param {number} bytes - 字节数
     * @returns {string} 格式化后的大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    },

    /**
     * 格式化数字
     * @param {number} num - 数字
     * @param {number} decimals - 小数位数
     * @returns {string} 格式化后的数字
     */
    formatNumber(num, decimals = 2) {
        return Number(num).toFixed(decimals);
    },

    /**
     * 防抖函数
     * @param {Function} func - 要防抖的函数
     * @param {number} wait - 等待时间(毫秒)
     * @returns {Function} 防抖后的函数
     */
    debounce(func, wait = 300) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    },

    /**
     * 节流函数
     * @param {Function} func - 要节流的函数
     * @param {number} limit - 时间限制(毫秒)
     * @returns {Function} 节流后的函数
     */
    throttle(func, limit = 300) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * 复制文本到剪贴板
     * @param {string} text - 要复制的文本
     * @returns {Promise<boolean>} 是否成功
     */
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showMessage('已复制到剪贴板', 'success');
            return true;
        } catch (err) {
            console.error('复制失败:', err);
            this.showMessage('复制失败', 'error');
            return false;
        }
    },

    /**
     * 下载文件
     * @param {string} url - 文件URL
     * @param {string} filename - 文件名
     */
    downloadFile(url, filename) {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    },

    /**
     * 生成唯一ID
     * @returns {string} 唯一ID
     */
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
};
