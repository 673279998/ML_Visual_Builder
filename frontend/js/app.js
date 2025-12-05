/**
 * 机器学习可视化平台 - 主应用脚本
 */

console.log('ML Visual Builder v1.0.0 已加载');

// 应用主对象
const MLApp = {
    init() {
        console.log('初始化应用...');
        this.setupNavigation();
        this.setupEventListeners();
    },

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
            if (!href) return;
            
            if (href === '#' || href.startsWith('javascript:')) {
                e.preventDefault();
                return;
            }
                
                // 对于普通页面链接，允许默认跳转
                // 可以在这里添加页面切换前的逻辑，如保存状态等
                const page = link.getAttribute('href');
                console.log('跳转到页面:', page);
            });
        });
        
        // 根据当前URL设置active状态
        this.setActiveNavLink();
    },
    
    setActiveNavLink() {
        const currentPath = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            const href = link.getAttribute('href');
            if (href === currentPath || (currentPath === 'index.html' && href === 'index.html')) {
                link.classList.add('active');
            }
        });
    },

    setupEventListeners() {
        // 保存按钮和执行按钮的事件监听已移至 WorkflowManager 中统一处理，避免重复绑定
        /*
        // 保存按钮
        const btnSave = document.getElementById('btn-save');
        if (btnSave) {
            btnSave.addEventListener('click', async () => {
                console.log('保存工作流');
                await this.saveWorkflow();
            });
        }

        // 绑定执行按钮
        /*
        const btnExecute = document.getElementById('btn-execute');
        if (btnExecute) {
            btnExecute.addEventListener('click', async () => {
                console.log('执行工作流');
                await this.executeWorkflow();
            });
        }
        */
        
        // 绑定自动适应按钮
        const btnAutoFit = document.getElementById('btn-auto-fit');
        if (btnAutoFit) {
            btnAutoFit.addEventListener('click', () => {
                if (window.workflowCanvas) {
                    window.workflowCanvas.autoFit();
                }
            });
        }
    },

    async saveWorkflow() {
        if (!window.workflowCanvas) {
            UIHelper.showMessage('工作流画布未初始化', 'error');
            return;
        }

        const workflowData = window.workflowCanvas.getWorkflowData();
        if (!workflowData.nodes || workflowData.nodes.length === 0) {
            UIHelper.showMessage('工作流为空，无法保存', 'warning');
            return;
        }

        // 显示保存对话框
        const name = prompt('请输入工作流名称:', '我的工作流');
        if (!name) return;

        const closeLoading = UIHelper.showLoading('保存中...');

        try {
            const result = await apiClient.createWorkflow({
                name: name,
                workflow_type: 'training',
                description: '通过画布创建的工作流',
                configuration: workflowData
            });

            if (result.success) {
                UIHelper.showMessage('工作流保存成功', 'success');
                this.currentWorkflowId = result.data.workflow_id;
            } else {
                throw new Error(result.error || '保存失败');
            }
        } catch (error) {
            console.error('保存工作流失败:', error);
            UIHelper.showMessage('保存失败: ' + error.message, 'error');
        } finally {
            closeLoading();
        }
    },

    async executeWorkflow() {
        if (!window.workflowCanvas) {
            UIHelper.showMessage('工作流画布未初始化', 'error');
            return;
        }

        const workflowData = window.workflowCanvas.getWorkflowData();
        if (!workflowData.nodes || workflowData.nodes.length === 0) {
            UIHelper.showMessage('工作流为空，无法执行', 'warning');
            return;
        }

        // 验证工作流是否完整
        const validation = this.validateWorkflow(workflowData);
        if (!validation.valid) {
            UIHelper.showMessage('工作流验证失败: ' + validation.error, 'error');
            return;
        }

        UIHelper.showMessage('工作流开始执行...', 'info');
        // 具体执行逻辑由 WorkflowManager 处理
    },

    validateWorkflow(workflowData) {
        // 简单验证，检查是否有数据源和算法节点
        const hasDataSource = workflowData.nodes.some(node => 
            node.name === '数据上传' || node.name === '数据集选择'
        );

        if (!hasDataSource) {
            return { valid: false, error: '缺少数据源节点' };
        }

        const hasAlgorithm = workflowData.nodes.some(node => 
            node.name.includes('算法')
        );

        if (!hasAlgorithm) {
            return { valid: false, error: '缺少算法节点' };
        }

        return { valid: true };
    },

    showMessage(message, type = 'info') {
        // 使用 UIHelper 显示消息
        UIHelper.showMessage(message, type);
    },

    currentWorkflowId: null
};

// DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    MLApp.init();
});
