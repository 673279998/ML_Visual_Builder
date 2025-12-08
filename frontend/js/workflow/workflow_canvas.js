/**
 * 工作流画布系统 - 核心功能
 * 使用Fabric.js实现拖拽和连接
 */

class WorkflowCanvas {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.canvas = null;
        this.nodes = new Map();  // 节点Map: id -> node对象
        this.connections = new Map();  // 连接Map: id -> connection对象
        this.selectedNode = null;
        this.isDraggingConnection = false;
        this.connectionStartNode = null;
        this.draggingPortType = null;
        this.tempConnection = null;
        this.nodeIdCounter = 1;
        this.connectionIdCounter = 1;
        
        this.init();
    }
    
    init() {
        // 初始化Fabric画布
        const container = document.getElementById(this.canvasId);
        if (!container) {
            console.error('画布容器不存在');
            return;
        }
        
        // 创建canvas元素
        const canvasEl = document.createElement('canvas');
        canvasEl.id = 'fabric-canvas';
        container.innerHTML = '';
        container.appendChild(canvasEl);
        
        // 计算画布实际可用空间
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        // 同时设置canvas的HTML属性和Fabric的尺寸
        canvasEl.width = containerWidth;
        canvasEl.height = containerHeight;
        
        this.canvas = new fabric.Canvas('fabric-canvas', {
            backgroundColor: '#f8f9fa',
            selection: false
        });
        
        // 绑定事件
        this.bindEvents();
        
        // 窗口resize时更新画布大小
        window.addEventListener('resize', () => this.resize());
    }
    
    bindEvents() {
        // 模块拖拽事件
        this.bindModuleDrag();
        
        // 画布点击事件
        this.canvas.on('mouse:down', (e) => this.onCanvasMouseDown(e));
        this.canvas.on('mouse:move', (e) => this.onCanvasMouseMove(e));
        this.canvas.on('mouse:up', (e) => this.onCanvasMouseUp(e));
        
        // 对象选择事件
        this.canvas.on('selection:created', (e) => this.onObjectSelected(e));
        this.canvas.on('selection:updated', (e) => this.onObjectSelected(e));
        this.canvas.on('selection:cleared', () => this.onSelectionCleared());
        
        // 对象移动事件 - 实时更新连接线
        this.canvas.on('object:moving', () => this.updateConnections());
        
        // 鼠标滚轮缩放
        this.canvas.on('mouse:wheel', (opt) => {
            const delta = opt.e.deltaY;
            let zoom = this.canvas.getZoom();
            zoom *= 0.999 ** delta;
            if (zoom > 20) zoom = 20;
            if (zoom < 0.1) zoom = 0.1;
            this.canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
            opt.e.preventDefault();
            opt.e.stopPropagation();
        });
        
        // 键盘事件
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
    }
    
    /**
     * 绑定左侧模块的拖拽事件
     */
    bindModuleDrag() {
        const moduleItems = document.querySelectorAll('.module-item');
        
        console.log('绑定拖拽事件，找到模块:', moduleItems.length);
        
        moduleItems.forEach(item => {
            const name = item.querySelector('.module-name')?.textContent || '';
            const icon = item.querySelector('.module-icon')?.textContent || '📦';
            
            item.addEventListener('dragstart', (e) => {
                console.log('开始拖拽:', { name, icon });
                e.dataTransfer.setData('module-name', name);
                e.dataTransfer.setData('module-icon', icon);
            });
        });
        
        // 画布作为拖放目标
        const container = document.getElementById(this.canvasId);
        
        if (container) {
            container.addEventListener('dragover', (e) => {
                e.preventDefault();
            });
            
            container.addEventListener('drop', (e) => {
                e.preventDefault();
                
                const moduleName = e.dataTransfer.getData('module-name');
                const moduleIcon = e.dataTransfer.getData('module-icon');
                
                console.log('拖拽放置:', { moduleName, moduleIcon });
                
                if (moduleName) {
                    // 计算相对于画布的坐标
                    const rect = container.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    
                    console.log('添加节点:', { moduleName, x, y });
                    this.addNode(moduleName, moduleIcon, x, y);
                }
            });
        }
    }
    
    /**
     * 添加节点到画布
     */
    addNode(name, icon, x, y) {
        const nodeId = 'node_' + this.nodeIdCounter++;
        
        // 创建节点组
        const nodeWidth = 180;
        const nodeHeight = 100;
        
        // 背景矩形 - 按钮样式
        const rect = new fabric.Rect({
            width: nodeWidth,
            height: nodeHeight,
            fill: '#ffffff',
            stroke: '#2196F3',
            strokeWidth: 2,
            rx: 12,
            ry: 12,
            shadow: 'rgba(0,0,0,0.1) 0px 2px 8px',
            strokeUniform: true  // 保持边框宽度一致
        });
        
        // 图标文本
        const iconText = new fabric.Text(icon, {
            fontSize: 32,
            top: 15,
            left: nodeWidth / 2,
            originX: 'center',
            originY: 'top',
            selectable: false
        });
        
        // 名称文本
        const nameText = new fabric.Text(name, {
            fontSize: 14,
            top: 55,
            left: nodeWidth / 2,
            originX: 'center',
            originY: 'top',
            selectable: false,
            fontWeight: 'bold'
        });
        
        // 状态文本
        const statusText = new fabric.Text('未配置', {
            fontSize: 12,
            top: 75,
            left: nodeWidth / 2,
            originX: 'center',
            originY: 'top',
            selectable: false,
            fill: '#999'
        });
        
        // 创建组
        const group = new fabric.Group([rect, iconText, nameText, statusText], {
            left: x - nodeWidth / 2,
            top: y - nodeHeight / 2,
            selectable: true,
            hasControls: false,
            hasBorders: true,
            lockScalingX: true,
            lockScalingY: true,
            hoverCursor: 'pointer',
            borderColor: '#4CAF50',  // 选中时边框颜色
            cornerColor: '#4CAF50',  // 选中时角点颜色
            transparentCorners: false,  // 显示选中角点
            borderScaleFactor: 2,  // 增加选中边框粗细
            className: 'enhanced-node workflow-node',  // 添加CSS类
            subTargetCheck: true  // 允许子对象接收事件，修复端口点击问题
        });
        
        // 存储节点数据
        group.nodeId = nodeId;
        group.nodeName = name;
        group.nodeConfig = {};
        group.nodeStatus = 'unconfigured';
        
        // 添加到画布
        this.canvas.add(group);
        this.nodes.set(nodeId, group);
        
        // 添加输入输出端口
        this.addPorts(group);
        
        // 双击编辑 - 使用fabric的事件
        group.on('mousedblclick', (e) => {
            console.log('双击节点:', nodeId, name);
            e.stopPropagation();
            this.showNodeConfig(nodeId);
        });
        
        // 节点悬停效果 - 仅记录日志
        group.on('mouseover', (e) => {
            console.log(`鼠标悬停节点: ${name}`);
        });

        group.on('mouseout', (e) => {
            console.log(`鼠标离开节点: ${name}`);
        });

        // 节点点击事件 - 用于选中
        group.on('mousedown', (e) => {
            console.log(`节点被点击: ${name} (${nodeId})`);
            
            // 清除之前选中节点的样式
            this.nodes.forEach((otherNode, otherId) => {
                if (otherId !== nodeId) {
                    otherNode.set({
                        borderColor: '#4CAF50'
                    });
                }
            });
            
            // 设置当前节点为选中状态
            this.canvas.setActiveObject(group);
            
            // 点击时的视觉反馈
            group.set({
                borderColor: '#FF5722'  // 点击时边框变红
            });
            
            this.canvas.renderAll();
            
            setTimeout(() => {
                group.set({
                    borderColor: '#4CAF50'
                });
                this.canvas.renderAll();
            }, 150);
        });
        
        console.log(`节点创建完成: ${name} (${nodeId}) - 节点尺寸: ${nodeWidth}x${nodeHeight}`);
        
        return nodeId;
    }
    
    /**
     * 添加节点的输入输出端口
     */
    addPorts(node) {
        const nodeWidth = 180;
        const nodeHeight = 100;
        const portRadius = 8;  // 端口半径

        // 输入端口（左侧边缘，垂直居中）
        // Group坐标系以中心为原点 (0,0)
        // 左边缘 x = -nodeWidth/2 = -90
        // 垂直居中 y = 0
        const inputPort = new fabric.Circle({
            radius: portRadius,
            fill: '#4CAF50',
            stroke: '#ffffff',
            strokeWidth: 2,
            left: -nodeWidth / 2 - portRadius,  // -90 - 8 = -98
            top: -portRadius,  // -8 (垂直居中)
            selectable: false,
            evented: true,
            hoverCursor: 'pointer',
            shadow: '0 1px 3px rgba(76, 175, 80, 0.5)',  // 绿色阴影
            opacity: 1,  // 完全不透明
            originX: 'left',
            originY: 'top'
        });
        inputPort.isPort = true;
        inputPort.portType = 'input';
        inputPort.parentNode = node;

        // 输出端口（右侧边缘，垂直居中）
        // 右边缘 x = nodeWidth/2 = 90
        const outputPort = new fabric.Circle({
            radius: portRadius,
            fill: '#FF9800',
            stroke: '#ffffff',
            strokeWidth: 2,
            left: nodeWidth / 2 - portRadius,  // 90 - 8 = 82
            top: -portRadius,  // -8 (垂直居中)
            selectable: false,
            evented: true,
            hoverCursor: 'pointer',
            shadow: '0 1px 3px rgba(255, 152, 0, 0.5)',  // 橙色阴影
            opacity: 1,  // 完全不透明
            originX: 'left',
            originY: 'top'
        });
        outputPort.isPort = true;
        outputPort.portType = 'output';
        outputPort.parentNode = node;
        
        node.inputPort = inputPort;
        node.outputPort = outputPort;
        
        // 使用正确的方法添加端口到组
        node.add(inputPort);
        node.add(outputPort);
        
        // 添加端口事件监听器
        inputPort.on('mousedown', (opt) => {
            console.log('输入端口被点击:', node.nodeName);
            if (opt.e) {
                opt.e.stopPropagation();
                opt.e.preventDefault();
            }
            
            // 临时锁定节点移动，防止拖拽端口时移动节点
            node.lockMovementX = true;
            node.lockMovementY = true;
            
            // 端口点击时的视觉反馈
            this.highlightPort(inputPort, true);
            setTimeout(() => {
                this.highlightPort(inputPort, false);
                // 延迟解锁，确保点击操作完成
                node.lockMovementX = false;
                node.lockMovementY = false;
            }, 200);
        });

        // 添加端口悬停效果
        inputPort.on('mouseover', (e) => {
            inputPort.set({
                scaleX: 1.3,
                scaleY: 1.3,
                shadow: '0 2px 6px rgba(76, 175, 80, 0.7)'
            });
            this.canvas.renderAll();
        });

        inputPort.on('mouseout', (e) => {
            inputPort.set({
                scaleX: 1.0,
                scaleY: 1.0,
                shadow: '0 1px 3px rgba(76, 175, 80, 0.5)'
            });
            this.canvas.renderAll();
        });
        
        outputPort.on('mousedown', (opt) => {
            console.log('输出端口被点击:', node.nodeName);
            if (opt.e) {
                opt.e.stopPropagation();
                opt.e.preventDefault();
            }
            
            // 锁定节点移动，防止拖拽端口时移动节点
            node.lockMovementX = true;
            node.lockMovementY = true;
            
            this.onPortClick('output', node, opt);
        });

        // 添加端口悬停效果
        outputPort.on('mouseover', (e) => {
            outputPort.set({
                scaleX: 1.3,
                scaleY: 1.3,
                shadow: '0 2px 6px rgba(255, 152, 0, 0.7)'
            });
            this.canvas.renderAll();
        });

        outputPort.on('mouseout', (e) => {
            outputPort.set({
                scaleX: 1.0,
                scaleY: 1.0,
                shadow: '0 1px 3px rgba(255, 152, 0, 0.5)'
            });
            this.canvas.renderAll();
        });
        
        console.log(`端口添加完成: ${node.nodeName}`, {
            inputPort: {
                left: inputPort.left,
                top: inputPort.top,
                radius: inputPort.radius,
                color: '#4CAF50'
            },
            outputPort: {
                left: outputPort.left,
                top: outputPort.top,
                radius: outputPort.radius,
                color: '#FF9800'
            },
            nodeSize: { width: nodeWidth, height: nodeHeight }
        });
    }
    
    /**
     * 查找两个节点间是否已有连接
     */
    findConnection(fromNode, toNode) {
        for (const [connId, line] of this.connections) {
            if (line.fromNode === fromNode && line.toNode === toNode) {
                return line;
            }
        }
        return null;
    }
    
    /**
     * 创建连接线
     */
    createConnection(fromNode, toNode) {
        const connectionId = 'conn_' + this.connectionIdCounter++;
        
        // 计算起点和终点
        const fromPoint = this.getPortPosition(fromNode, 'output');
        const toPoint = this.getPortPosition(toNode, 'input');
        
        console.log('创建连接:', {
            from: fromPoint,
            to: toPoint,
            fromNode: fromNode.nodeName,
            toNode: toNode.nodeName
        });
        
        // 创建直线（更简单可靠）
        const line = new fabric.Line(
            [fromPoint.x, fromPoint.y, toPoint.x, toPoint.y],
            {
                stroke: '#2196F3',
                strokeWidth: 2,
                selectable: true,     // 允许选中
                evented: true,        // 允许事件
                hasControls: false,   // 无控制器
                hasBorders: false,    // 无边框
                lockMovementX: true,  // 锁定移动
                lockMovementY: true,  // 锁定移动
                perPixelTargetFind: false,
                targetFindTolerance: 4, // 增加点击容差
                strokeDashArray: [0]  // 实线
            }
        );
        
        line.connectionId = connectionId;
        line.fromNode = fromNode;
        line.toNode = toNode;
        
        this.canvas.add(line);
        this.connections.set(connectionId, line);
        
        // 移到底层
        line.sendToBack();
        
        console.log('连接创建成功:', connectionId);
        
        return connectionId;
    }
    
    /**
     * 获取端口的全局坐标
     */
    getPortPosition(node, portType) {
        const port = portType === 'input' ? node.inputPort : node.outputPort;
        if (!port || !node) return { x: 0, y: 0 };
        
        // 计算 Group 的中心点坐标
        // 注意：node.left/top 是 Group 左上角的坐标（默认 originX/Y 为 left/top）
        // node.width/height 是 Group 的尺寸
        // 需要加上 width/2 和 height/2 得到中心点
        const groupCenterX = node.left + (node.width * node.scaleX) / 2;
        const groupCenterY = node.top + (node.height * node.scaleY) / 2;
        
        // 端口坐标是相对于 Group 中心的
        // port.left 是端口左上角相对于 Group 中心的 X 偏移
        // 我们需要端口中心的全局坐标
        // 端口中心相对 X = port.left + port.radius
        // 端口中心相对 Y = port.top + port.radius
        const portCenterX = groupCenterX + (port.left + port.radius) * node.scaleX;
        const portCenterY = groupCenterY + (port.top + port.radius) * node.scaleY;
        
        console.log(`端口位置计算: ${portType}`, {
            portLeft: port.left,
            portTop: port.top,
            groupCenter: { x: groupCenterX, y: groupCenterY },
            result: { x: portCenterX, y: portCenterY }
        });
        
        return {
            x: portCenterX,
            y: portCenterY
        };
    }
    
    /**
     * 生成连接线路径
     */
    getConnectionPath(from, to) {
        const dx = to.x - from.x;
        const cp1x = from.x + dx / 3;
        const cp2x = to.x - dx / 3;
        
        return `M ${from.x} ${from.y} C ${cp1x} ${from.y}, ${cp2x} ${to.y}, ${to.x} ${to.y}`;
    }
    
    /**
     * 更新连接线位置
     */
    updateConnections() {
        this.connections.forEach(line => {
            if (!line.fromNode || !line.toNode) return;
            
            const fromPoint = this.getPortPosition(line.fromNode, 'output');
            const toPoint = this.getPortPosition(line.toNode, 'input');
            
            // 更新直线端点
            line.set({
                x1: fromPoint.x,
                y1: fromPoint.y,
                x2: toPoint.x,
                y2: toPoint.y
            });
            line.setCoords();
        });
        this.canvas.renderAll();
    }
    
    /**
     * 端口点击事件处理
     */
    onPortClick(portType, node, event) {
        console.log(`端口点击: ${portType} - ${node.nodeName}`);
        
        if (portType === 'output') {
            // 开始拖拽连接
            this.isDraggingConnection = true;
            this.connectionStartNode = node;
            this.draggingPortType = 'output';
            
            // 创建临时连接线
            const startPoint = this.getPortPosition(node, 'output');
            this.tempConnection = new fabric.Line(
                [startPoint.x, startPoint.y, startPoint.x, startPoint.y],
                {
                    stroke: '#2196F3',
                    strokeWidth: 3,
                    strokeDashArray: [8, 4],
                    selectable: false,
                    evented: false
                }
            );
            this.canvas.add(this.tempConnection);
            this.tempConnection.bringToFront();
            
            console.log('开始拖拽连接从:', node.nodeName);
        }
    }
    
    /**
     * 画布鼠标按下事件
     */
    onCanvasMouseDown(opt) {
        if (!opt.target) {
            // 点击了空白区域，清除所有选择
            console.log('点击空白区域，清除选择');
            this.canvas.discardActiveObject();
            this.onSelectionCleared();
            this.canvas.renderAll();
            return;
        }
        
        // 检查是否点击了端口
        // 注意：由于 Group 设置了 subTargetCheck: true，opt.target 可能是端口对象
        if (opt.target.isPort) {
            // 阻止事件冒泡和默认行为
            if (opt.e) {
                opt.e.stopPropagation();
                opt.e.preventDefault();
            }
            
            // 关键：取消当前选中的对象（即包含该端口的 Group），防止节点跟随移动
            this.canvas.discardActiveObject();
            this.canvas.requestRenderAll();
            
            this.onPortClick(opt.target.portType, opt.target.parentNode, opt);
            return;
        }
        
        // 清除拖拽连接状态
        this.isDraggingConnection = false;
        this.connectionStartNode = null;
        this.draggingPortType = null;
        if (this.tempConnection) {
            this.canvas.remove(this.tempConnection);
            this.tempConnection = null;
        }
    }
    
    /**
     * 画布鼠标移动事件
     */
    onCanvasMouseMove(opt) {
        if (this.isDraggingConnection && this.tempConnection) {
            const pointer = this.canvas.getPointer(opt.e);
            this.tempConnection.set({
                x2: pointer.x,
                y2: pointer.y
            });
            this.tempConnection.setCoords();
            this.canvas.renderAll();
            
            // 检查鼠标是否悬停在输入端口上
            // 注意：由于 Group 设置了 subTargetCheck: true，opt.target 可能是端口对象
            const target = opt.target;
            if (target && target.isPort && target.portType === 'input' && target.parentNode !== this.connectionStartNode) {
                // 高亮目标输入端口
                this.highlightPort(target, true);
                this.tempConnection.stroke = '#4CAF50';  // 绿色表示可以连接
            } else {
                // 取消高亮
                // 注意：这里可能需要遍历所有端口来取消高亮，或者记录上一个高亮的端口
                // 目前的实现依赖于 mouseout 事件，但在拖拽过程中 mouseout 可能不会触发
                // 暂时保持现状，如果发现高亮不消失再修复
                this.tempConnection.stroke = '#2196F3';  // 恢复蓝色
            }
        }
    }
    
    /**
     * 高亮端口
     */
    highlightPort(port, highlight) {
        if (!port) return;
        
        if (highlight) {
            port.set({
                stroke: '#ffffff',
                strokeWidth: 6,
                shadow: '0 4px 8px rgba(76, 175, 80, 0.5)'
            });
        } else {
            port.set({
                stroke: '#ffffff',
                strokeWidth: 4,
                shadow: '0 2px 4px rgba(0,0,0,0.3)'
            });
        }
        this.canvas.renderAll();
    }
    
    /**
     * 画布鼠标松开事件
     */
    onCanvasMouseUp(opt) {
        // 无论如何，鼠标松开时都要解锁当前操作节点的移动
        if (this.connectionStartNode) {
            this.connectionStartNode.lockMovementX = false;
            this.connectionStartNode.lockMovementY = false;
        }

        if (this.isDraggingConnection && this.connectionStartNode) {
            let connected = false;
            let targetNode = null;
            
            // 1. 首先尝试直接从事件目标获取
            if (opt.target && opt.target.isPort && opt.target.portType === 'input') {
                targetNode = opt.target.parentNode;
            } 
            // 2. 如果事件目标不是端口（可能是Group或其他），则手动进行碰撞检测
            else {
                // 获取鼠标全局坐标
                const pointer = this.canvas.getPointer(opt.e);
                
                // 遍历所有节点，检查鼠标是否在某个输入端口范围内
                for (const [nodeId, node] of this.nodes) {
                    // 跳过起始节点
                    if (node === this.connectionStartNode) continue;
                    
                    // 获取该节点的输入端口位置
                    const portPos = this.getPortPosition(node, 'input');
                    // 端口半径（增加一点容差，更容易选中）
                    const portRadius = 12; 
                    
                    // 计算距离
                    const dist = Math.sqrt(
                        Math.pow(pointer.x - portPos.x, 2) + 
                        Math.pow(pointer.y - portPos.y, 2)
                    );
                    
                    if (dist <= portRadius) {
                        targetNode = node;
                        break;
                    }
                }
            }
            
            // 如果找到了目标节点，尝试建立连接
            if (targetNode) {
                // 确保不是同一个节点，并且不存在重复连接
                if (targetNode !== this.connectionStartNode) {
                    const existingConnection = this.findConnection(this.connectionStartNode, targetNode);
                    if (!existingConnection) {
                        this.createConnection(this.connectionStartNode, targetNode);
                        connected = true;
                        console.log('✅ 连接创建成功:', this.connectionStartNode.nodeName, '->', targetNode.nodeName);
                        
                        // 显示成功消息
                        if (window.UIHelper) {
                            UIHelper.showMessage(`连接成功: ${this.connectionStartNode.nodeName} → ${targetNode.nodeName}`, 'success');
                        }
                    } else {
                        console.log('⚠️ 连接已存在:', this.connectionStartNode.nodeName, '->', targetNode.nodeName);
                        
                        if (window.UIHelper) {
                            UIHelper.showMessage('连接已存在', 'warning');
                        }
                    }
                } else {
                    console.log('⚠️ 不能连接到自己:', this.connectionStartNode.nodeName);
                    
                    if (window.UIHelper) {
                        UIHelper.showMessage('不能连接到自己', 'warning');
                    }
                }
            } else {
                console.log('❌ 连接取消 - 未找到目标端口');
            }
            
            // 清理临时连接线
            if (this.tempConnection) {
                this.canvas.remove(this.tempConnection);
                this.tempConnection = null;
            }
            
            // 清理状态
            this.isDraggingConnection = false;
            this.connectionStartNode = null;
            this.draggingPortType = null;
            
            this.canvas.renderAll();
        }
    }
    
    /**
     * 对象选中事件
     */
    onObjectSelected(e) {
        const obj = e.selected[0];
        if (obj) {
            if (obj.nodeId) {
                this.selectedNode = obj;
                this.showNodeProperties(obj.nodeId);
            } else if (obj.connectionId) {
                // 选中连接线，高亮显示
                obj.set({
                    stroke: '#FF5722',
                    strokeWidth: 4
                });
                this.canvas.renderAll();
            }
        }
    }
    
    /**
     * 选择清除事件
     */
    onSelectionCleared() {
        console.log('清除了选择');
        
        // 清除所有节点的选中状态样式
        this.nodes.forEach((node, nodeId) => {
            if (node.originalScaleX !== undefined) {
                node.set({
                    scaleX: node.originalScaleX,
                    scaleY: node.originalScaleY,
                    shadow: node.originalShadow || 'rgba(0,0,0,0.1) 0px 2px 8px',
                    borderColor: '#4CAF50'
                });
            }
        });
        
        // 清除所有连接线的选中状态
        this.connections.forEach(line => {
            line.set({
                stroke: '#2196F3',
                strokeWidth: 2
            });
        });
        
        this.selectedNode = null;
        this.canvas.renderAll();  // 刷新画布
        this.hideNodeProperties();
    }
    
    /**
     * 键盘事件
     */
    onKeyDown(e) {
        // 如果当前焦点在输入框中，不触发删除
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
            return;
        }

        // Delete或Backspace键删除选中对象
        if (e.key === 'Delete' || e.key === 'Backspace') {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj) {
                if (activeObj.nodeId) {
                    this.deleteNode(activeObj.nodeId);
                } else if (activeObj.connectionId) {
                    this.deleteConnection(activeObj.connectionId);
                }
            }
        }
    }
    
    /**
     * 删除连接
     */
    deleteConnection(connId) {
        const line = this.connections.get(connId);
        if (line) {
            this.canvas.remove(line);
            this.connections.delete(connId);
            console.log('删除连接:', connId);
            this.canvas.renderAll();
        }
    }
    
    /**
     * 删除节点
     */
    deleteNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        // 删除相关连接
        const connectionsToDelete = [];
        this.connections.forEach((line, connId) => {
            if (line.fromNode === node || line.toNode === node) {
                connectionsToDelete.push(connId);
            }
        });
        
        connectionsToDelete.forEach(connId => {
            const line = this.connections.get(connId);
            this.canvas.remove(line);
            this.connections.delete(connId);
        });
        
        // 删除节点
        this.canvas.remove(node);
        this.nodes.delete(nodeId);
        
        this.selectedNode = null;
        this.hideNodeProperties();
    }
    
    /**
     * 显示节点配置面板
     */
    showNodeConfig(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        // 触发自定义事件，由外部处理
        const event = new CustomEvent('node-config', {
            detail: {
                nodeId: nodeId,
                nodeName: node.nodeName,
                config: node.nodeConfig
            }
        });
        document.dispatchEvent(event);
    }
    
    /**
     * 显示节点属性面板
     */
    showNodeProperties(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        const panel = document.querySelector('.properties-content');
        if (!panel) return;
        
        // 如果是模型结果或可视化节点
            if (node.nodeName === '模型结果' || node.nodeName === '可视化') {
                if (node.executionResult) {
                    this.showResultsInPanel(panel, node, nodeId);
                    return;
                } else if (node.nodeStatus === 'success') {
                    // 状态为成功但无结果，可能是数据丢失或加载失败
                    panel.innerHTML = `
                        <div style="padding: 20px; text-align: center;">
                            <h3 style="color: #ff9800; margin-bottom: 10px;">暂无结果数据</h3>
                            <p style="color: #666; margin-bottom: 15px;">节点已运行完成，但未找到可显示的结果。</p>
                            <div style="font-size: 12px; color: #999; background: #f5f5f5; padding: 10px; border-radius: 4px; text-align: left;">
                                可能原因：<br>
                                1. 模型训练未产生有效指标<br>
                                2. 数据传输过程中丢失（建议重新运行工作流）<br>
                                3. 结果格式不兼容
                            </div>
                        </div>
                        <div class="property-section" style="padding: 0 20px 20px;">
                            <button class="btn btn-primary btn-block" data-node-config="${nodeId}">
                                配置节点
                            </button>
                            <button class="btn btn-secondary btn-block" data-node-delete="${nodeId}">
                                删除节点
                            </button>
                        </div>
                    `;
                    
                    // 绑定事件监听器
                    const configBtn = panel.querySelector(`[data-node-config="${nodeId}"]`);
                    const deleteBtn = panel.querySelector(`[data-node-delete="${nodeId}"]`);
                    
                    if (configBtn) {
                        configBtn.addEventListener('click', () => {
                            console.log('配置节点:', nodeId);
                            this.showNodeConfig(nodeId);
                        });
                    }
                    
                    if (deleteBtn) {
                        deleteBtn.addEventListener('click', () => {
                            console.log('删除节点:', nodeId);
                            this.deleteNode(nodeId);
                        });
                    }
                    return;
                }
            }
        
        // 其他节点显示基本信息
        panel.innerHTML = `
            <div class="property-section">
                <h4>节点信息</h4>
                <div class="property-item">
                    <label>节点ID:</label>
                    <span>${nodeId}</span>
                </div>
                <div class="property-item">
                    <label>节点类型:</label>
                    <span>${node.nodeName}</span>
                </div>
                <div class="property-item">
                    <label>状态:</label>
                    <span class="status-badge status-${node.nodeStatus}">${this.getStatusText(node.nodeStatus)}</span>
                </div>
            </div>
            <div class="property-section">
                <button class="btn btn-primary btn-block" data-node-config="${nodeId}">
                    配置节点
                </button>
                <button class="btn btn-secondary btn-block" data-node-delete="${nodeId}">
                    删除节点
                </button>
            </div>
        `;
        
        // 绑定事件监听器
        const configBtn = panel.querySelector(`[data-node-config="${nodeId}"]`);
        const deleteBtn = panel.querySelector(`[data-node-delete="${nodeId}"]`);
        
        if (configBtn) {
            configBtn.addEventListener('click', () => {
                console.log('配置节点:', nodeId);
                this.showNodeConfig(nodeId);
            });
        }
        
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                console.log('删除节点:', nodeId);
                this.deleteNode(nodeId);
            });
        }
    }
    
    /**
     * 在属性面板中显示结果
     */
    showResultsInPanel(panel, node, nodeId) {
        const result = node.executionResult;
        
        console.log('显示结果到属性面板:', { nodeId, nodeName: node.nodeName, result });
        
        // 清空面板
        panel.innerHTML = '';
        panel.scrollTop = 0;
        panel.style.overflowY = 'auto';
        panel.style.maxHeight = 'calc(100vh - 150px)';
        
        // 添加标题
        const header = document.createElement('div');
        const algorithmName = result.algorithm_display_name || result.algorithm_name || result.algorithm || 'N/A';
        header.style.cssText = 'padding: 15px; background: #f5f5f5; border-bottom: 2px solid #2196F3; position: sticky; top: 0; z-index: 10;';
        header.innerHTML = `
            <h3 style="margin: 0; color: #333; font-size: 16px;">
                ${node.nodeName === '模型结果' ? '模型训练结果' : '可视化结果'}
            </h3>
            <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
                节点ID: ${nodeId} | 算法: ${algorithmName}
            </p>
        `;
        panel.appendChild(header);
        
        // 创建结果容器
        const resultsContainer = document.createElement('div');
        resultsContainer.id = `results-${nodeId}`;
        resultsContainer.style.cssText = 'padding: 15px;';
        panel.appendChild(resultsContainer);
        
        // 检查ResultVisualizer是否可用
        if (!window.ResultVisualizer) {
            console.error('ResultVisualizer未加载');
            resultsContainer.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #f44336;">
                    <p><strong>错误: ResultVisualizer未加载</strong></p>
                    <p style="font-size: 12px; margin-top: 10px;">请检查index.html是否正确引入result_visualizer.js</p>
                    <details style="margin-top: 20px; text-align: left;">
                        <summary style="cursor: pointer; color: #666;">查看原始数据</summary>
                        <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 300px; overflow: auto; margin-top: 10px;">${JSON.stringify(result, null, 2)}</pre>
                    </details>
                </div>
            `;
            return;
        }
        
        // 使用ResultVisualizer渲染结果
        try {
            const visualizer = new ResultVisualizer();
            const algorithmType = result.algorithm_type || 'classification';
            
            if (node.nodeName === '模型结果') {
                console.log('渲染模型结果...');
                
                // 模型结果节点:显示指标和基本信息
                // 增强兼容性：优先从 complete_results 获取，其次从根对象获取
                const metrics = result.complete_results?.metrics || result.performance_metrics || result.metrics || {};
                console.log('指标数据:', metrics);
                
                if (Object.keys(metrics).length > 0) {
                    visualizer.renderMetrics(
                        resultsContainer, 
                        algorithmType,
                        metrics
                    );
                } else {
                    resultsContainer.innerHTML += '<p style="color: #999; padding: 10px;">没有可用的指标数据</p>';
                }
                
                // 如果有特征重要性，也显示出来
                if (result.feature_importance) {
                     // 简单显示特征重要性，或者可以调用 visualizer 的方法如果存在
                     // ResultVisualizer 可能没有单独的 renderFeatureImportance 方法暴露出来，
                     // 但我们可以检查一下。暂时先不加，以免报错。
                }
                
            } else {
                // 可视化节点:显示所有可视化图表
                const visualizations = result.complete_results?.visualizations || result.visualizations || {};
                
                if (Object.keys(visualizations).length === 0) {
                    resultsContainer.innerHTML += '<p style="color: #999; padding: 10px;">没有可用的可视化数据</p>';
                } else {
                    // 筛选最关键的一个可视化图表
                    let criticalVis = {};
                    let hasCritical = false;

                    if (algorithmType === 'classification') {
                        // 优先显示混淆矩阵，其次ROC曲线
                        if (visualizations.confusion_matrix) {
                            criticalVis.confusion_matrix = visualizations.confusion_matrix;
                            hasCritical = true;
                        } else if (visualizations.roc_curve) {
                            criticalVis.roc_curve = visualizations.roc_curve;
                            hasCritical = true;
                        }
                    } else if (algorithmType === 'regression') {
                        // 优先显示预测vs实际，其次残差图
                        if (visualizations.prediction_vs_actual) {
                            criticalVis.prediction_vs_actual = visualizations.prediction_vs_actual;
                            hasCritical = true;
                        } else if (visualizations.residuals) {
                            criticalVis.residuals = visualizations.residuals;
                            hasCritical = true;
                        }
                    } else if (algorithmType === 'clustering') {
                        // 优先显示散点图
                        if (visualizations.cluster_scatter) {
                            criticalVis.cluster_scatter = visualizations.cluster_scatter;
                            hasCritical = true;
                        } else if (visualizations.silhouette) {
                            criticalVis.silhouette = visualizations.silhouette;
                            hasCritical = true;
                        }
                    } else if (algorithmType === 'dimensionality_reduction') {
                        if (visualizations.pca_scatter) {
                            criticalVis.pca_scatter = visualizations.pca_scatter;
                            hasCritical = true;
                        } else if (visualizations.tsne_scatter) {
                            criticalVis.tsne_scatter = visualizations.tsne_scatter;
                            hasCritical = true;
                        }
                    }

                    // 如果没有找到定义的关键图表，默认取第一个
                    if (!hasCritical) {
                        const firstKey = Object.keys(visualizations)[0];
                        criticalVis[firstKey] = visualizations[firstKey];
                    }

                    if (algorithmType === 'classification') {
                        visualizer.renderClassificationVisualizations(resultsContainer, criticalVis);
                    } else if (algorithmType === 'regression') {
                        visualizer.renderRegressionVisualizations(resultsContainer, criticalVis);
                    } else if (algorithmType === 'clustering') {
                        visualizer.renderClusteringVisualizations(resultsContainer, criticalVis);
                    } else if (algorithmType === 'dimensionality_reduction') {
                        visualizer.renderDimensionalityReductionVisualizations(resultsContainer, criticalVis);
                    }
                }
            }
            
            // 添加详情链接提示
            const footer = document.createElement('div');
            footer.style.cssText = 'padding: 15px; text-align: center; color: #666; font-size: 12px; border-top: 1px solid #eee; margin-top: 20px;';
            footer.innerHTML = '详细的内容请参见模型详情';
            resultsContainer.appendChild(footer);
        } catch (error) {
            console.error('渲染结果失败:', error);
            resultsContainer.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #f44336;">
                    <p><strong>渲染错误</strong></p>
                    <p style="font-size: 12px; margin-top: 10px;">${error.message}</p>
                    <details style="margin-top: 20px; text-align: left;">
                        <summary style="cursor: pointer; color: #666;">查看错误详情</summary>
                        <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 300px; overflow: auto; margin-top: 10px;">${error.stack}</pre>
                    </details>
                </div>
            `;
        }
    }
    
    /**
     * 隐藏节点属性面板
     */
    hideNodeProperties() {
        const panel = document.querySelector('.properties-content');
        if (panel) {
            panel.innerHTML = '<p class="empty-hint">选择一个模块查看其属性</p>';
        }
    }

    
    
    /**
     * 获取状态文本
     */
    getStatusText(status) {
        const statusMap = {
            'unconfigured': '未配置',
            'configured': '已配置',
            'running': '运行中',
            'success': '成功',
            'error': '错误'
        };
        return statusMap[status] || status;
    }
    
    /**
     * 更新节点状态
     */
    updateNodeStatus(nodeId, status, statusText) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        node.nodeStatus = status;
        
        // 更新状态文本
        const items = node.getObjects();
        if (items.length >= 4) {
            items[3].set('text', statusText || this.getStatusText(status));
        }
        
        // 更新边框颜色
        const statusColors = {
            'unconfigured': '#999',
            'configured': '#2196F3',
            'running': '#FF9800',
            'success': '#4CAF50',
            'error': '#F44336'
        };
        items[0].set('stroke', statusColors[status] || '#2196F3');
        
        this.canvas.renderAll();
    }
    
    /**
     * 更新节点配置
     */
    updateNodeConfig(nodeId, config) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        node.nodeConfig = config;
        this.updateNodeStatus(nodeId, 'configured', '已配置');
    }
    
    /**
     * 自动适应画布
     * 将所有节点缩放并居中显示在画布内
     */
    autoFit() {
        if (this.nodes.size === 0) {
             this.canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
             return;
        }

        // 计算所有节点的边界框
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        this.nodes.forEach(node => {
            // node.left/top 是节点左上角坐标（在没有旋转的情况下）
            // 需要考虑缩放
            const width = node.width * node.scaleX;
            const height = node.height * node.scaleY;
            
            if (node.left < minX) minX = node.left;
            if (node.top < minY) minY = node.top;
            if (node.left + width > maxX) maxX = node.left + width;
            if (node.top + height > maxY) maxY = node.top + height;
        });
        
        // 如果计算结果无效，直接返回
        if (minX === Infinity || maxX === -Infinity) return;

        const padding = 50; // 边距
        const width = maxX - minX + padding * 2;
        const height = maxY - minY + padding * 2;
        
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        // 计算缩放比例
        const scaleX = canvasWidth / width;
        const scaleY = canvasHeight / height;
        let scale = Math.min(scaleX, scaleY);
        
        // 限制缩放范围，避免过度放大或缩小
        if (scale > 1) scale = 1; 
        if (scale < 0.1) scale = 0.1;
        
        // 计算中心点
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        
        // 计算平移量，使中心点对应画布中心
        const panX = (canvasWidth / 2) - (centerX * scale);
        const panY = (canvasHeight / 2) - (centerY * scale);
        
        // 应用变换
        this.canvas.setViewportTransform([scale, 0, 0, scale, panX, panY]);
        this.canvas.renderAll();
        
        console.log('自动适应画布:', { scale, panX, panY });
    }
    
    /**
     * 获取工作流数据
     */
    getWorkflowData() {
        const nodes = [];
        const connections = [];
        
        this.nodes.forEach((node, nodeId) => {
            nodes.push({
                id: nodeId,
                name: node.nodeName,
                x: node.left,
                y: node.top,
                config: node.nodeConfig,
                status: node.nodeStatus
            });
        });
        
        this.connections.forEach((line, connId) => {
            connections.push({
                id: connId,
                from: line.fromNode.nodeId,
                to: line.toNode.nodeId
            });
        });
        
        return { nodes, connections };
    }
    
    /**
     * 加载工作流数据
     */
    loadWorkflowData(data) {
        // 清空当前画布
        this.clear();
        
        // 加载节点
        data.nodes.forEach(nodeData => {
            const nodeId = this.addNode(nodeData.name, '📦', nodeData.x, nodeData.y);
            const node = this.nodes.get(nodeId);
            if (node) {
                node.nodeConfig = nodeData.config;
                // 更新节点状态（同时更新UI显示）
                if (nodeData.status) {
                    this.updateNodeStatus(nodeId, nodeData.status);
                }
            }
        });
        
        // 加载连接（需要等节点都创建完成）
        setTimeout(() => {
            data.connections.forEach(connData => {
                const fromNode = this.nodes.get(connData.from);
                const toNode = this.nodes.get(connData.to);
                if (fromNode && toNode) {
                    this.createConnection(fromNode, toNode);
                }
            });
        }, 100);
    }
    
    /**
     * 清空画布
     */
    clear() {
        this.canvas.clear();
        this.canvas.backgroundColor = '#f8f9fa';
        this.nodes.clear();
        this.connections.clear();
        this.selectedNode = null;
        this.nodeIdCounter = 1;
        this.connectionIdCounter = 1;
    }
    
    /**
     * 调整画布大小
     */
    resize() {
        const container = document.getElementById(this.canvasId);
        if (container && this.canvas) {
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            // 同时更新canvas HTML属性和Fabric尺寸
            const canvasEl = document.getElementById('fabric-canvas');
            if (canvasEl) {
                canvasEl.width = width;
                canvasEl.height = height;
            }
            
            this.canvas.setDimensions({ width, height });
            this.canvas.renderAll();
        }
    }
}

// 导出
window.WorkflowCanvas = WorkflowCanvas;
