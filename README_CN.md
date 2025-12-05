# ML Visual Builder - 机器学习可视化平台

## 🚀 项目概述

ML Visual Builder 是一个基于 Web 的机器学习可视化平台，通过拖拽式界面让用户无需编写代码即可构建、训练和部署机器学习模型。平台集成了数据预处理、特征工程、模型训练、超参数调优和结果可视化等完整机器学习流程。

## ✨ 核心功能

### 📊 数据管理
- **多格式支持**: CSV、Excel、JSON 等常见数据格式导入
- **数据预览**: 实时查看数据结构和统计信息
- **数据清洗**: 缺失值处理、异常值检测、数据标准化
- **特征工程**: 特征选择、特征变换、特征编码

### 🧩 可视化工作流
- **拖拽式构建**: 通过画布拖拽组件构建机器学习流程
- **节点化设计**: 数据节点、预处理节点、算法节点、评估节点
- **流程可视化**: 实时展示数据处理和模型训练流程
- **工作流管理**: 保存、加载、分享工作流配置

### 🤖 算法库

#### 分类算法
- **逻辑回归** (Logistic Regression) - 经典的线性分类模型
- **决策树分类器** (Decision Tree Classifier) - 基于树结构的分类算法
- **随机森林分类器** (Random Forest Classifier) - 集成学习，多棵决策树组合
- **支持向量机分类器** (SVM Classifier) - 基于最大间隔的分类算法
- **K近邻分类器** (KNN Classifier) - 基于距离度量的分类算法
- **朴素贝叶斯分类器** (Naive Bayes Classifier) - 基于贝叶斯定理的概率分类
- **梯度提升分类器** (Gradient Boosting Classifier) - 集成学习，逐步优化
- **多层感知器分类器** (MLP Classifier) - 神经网络分类器
- **XGBoost分类器** (XGBoost Classifier) - 优化的梯度提升算法
- **LightGBM分类器** (LightGBM Classifier) - 微软开发的梯度提升框架
- **CatBoost分类器** (CatBoost Classifier) - 处理类别特征的梯度提升算法

#### 回归算法
- **线性回归** (Linear Regression) - 基础的线性回归模型
- **岭回归** (Ridge Regression) - L2正则化的线性回归
- **Lasso回归** (Lasso Regression) - L1正则化的线性回归
- **弹性网络回归** (ElasticNet Regression) - L1+L2正则化的线性回归
- **决策树回归器** (Decision Tree Regressor) - 基于树结构的回归算法
- **随机森林回归器** (Random Forest Regressor) - 集成学习的回归算法
- **梯度提升回归器** (Gradient Boosting Regressor) - 集成学习的回归算法
- **支持向量回归器** (SVR) - 基于支持向量机的回归算法
- **多层感知器回归器** (MLP Regressor) - 神经网络回归器
- **XGBoost回归器** (XGBoost Regressor) - 优化的梯度提升回归
- **LightGBM回归器** (LightGBM Regressor) - 微软开发的梯度提升回归

#### 聚类算法
- **K均值聚类** (K-Means Clustering) - 基于距离的经典聚类算法
- **DBSCAN聚类** (DBSCAN Clustering) - 基于密度的聚类算法
- **高斯混合模型** (GMM Clustering) - 基于概率分布的聚类算法
- **层次聚类** (Hierarchical Clustering) - 构建聚类层次结构
- **谱聚类** (Spectral Clustering) - 基于图论的聚类算法

#### 降维算法
- **主成分分析** (PCA) - 线性降维，最大化方差
- **线性判别分析** (LDA) - 有监督的线性降维
- **t-SNE降维** (t-SNE) - 非线性降维，保持局部结构
- **UMAP降维** (UMAP) - 非线性降维，保持全局和局部结构

### ⚙️ 模型训练与调优
- **自动超参数调优**: 网格搜索、随机搜索、贝叶斯优化
- **交叉验证**: K折交叉验证、留一法验证
- **模型评估**: 准确率、精确率、召回率、F1分数、AUC、RMSE、MAE 等指标
- **模型持久化**: 训练好的模型可保存为文件或数据库存储

### 📈 结果可视化
- **训练过程可视化**: 损失曲线、准确率曲线、特征重要性
- **预测结果可视化**: 混淆矩阵、ROC曲线、残差图、聚类结果可视化
- **交互式图表**: 支持缩放、平移、数据点查看
- **报告生成**: 自动生成模型评估报告

## 🎯 项目特色

### 🎨 用户友好界面
- **零代码操作**: 无需编程经验，通过可视化界面完成机器学习任务
- **直观设计**: 简洁明了的界面设计，降低学习成本
- **实时反馈**: 操作过程中实时显示处理结果和状态

## 🚀 快速开始

### 环境要求
- Python 3.10 或更高版本
- Windows/macOS/Linux 操作系统
- 4GB 以上内存（推荐8GB）
- 2GB 以上可用磁盘空间

### 安装步骤

#### 方法一：使用启动脚本（推荐）
1. 克隆或下载项目到本地
2. 双击运行 `start.bat`（Windows）或执行 `./start.sh`（Linux/macOS）
3. 脚本会自动：
   - 检查 Python 环境
   - 安装 uv 包管理器
   - 创建虚拟环境
   - 安装所有依赖
   - 初始化数据库
   - 启动 Flask 服务器

#### 方法二：手动安装
```bash
# 1. 克隆项目
git clone https://github.com/673279998/ML_Visual_Builder.git
cd ML_Visual_Builder

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 初始化数据库
python backend/database/models.py

# 6. 启动服务器
cd backend
python app.py
```

### 运行平台
1. 启动成功后，在浏览器中访问：`http://localhost:5000`
2. 平台主界面包含以下功能模块：
   - **工作流**: 拖拽式构建机器学习流程
   - **数据管理**: 导入、查看、处理数据
   - **模型管理**: 查看已训练的模型
   - **模型预测**: 使用训练好的模型进行预测

## 📖 使用指南

### 1. 数据导入
- 点击"数据管理"页面
- 选择"导入数据"按钮
- 上传 CSV、Excel 或 JSON 文件
- 预览数据并确认导入

### 2. 创建工作流
- 进入"工作流"页面
- 从左侧工具栏拖拽组件到画布：
  - **数据源**: 选择已导入的数据集
  - **预处理**: 数据清洗、特征工程
  - **算法**: 选择机器学习算法
  - **评估**: 模型评估指标
- 连接组件构建完整流程

### 3. 模型训练
- 配置算法参数
- 设置训练/测试集划分比例
- 点击"执行"按钮开始训练
- 实时查看训练进度和结果

### 4. 结果分析
- 查看模型评估指标
- 分析可视化图表
- 保存训练好的模型
- 导出评估报告

## 🏗️ 项目结构

```
ML_Visual_Builder/
├── backend/                    # 后端代码
│   ├── algorithms/            # 算法实现
│   │   ├── classification/    # 分类算法
│   │   ├── regression/        # 回归算法
│   │   ├── clustering/        # 聚类算法
│   │   └── dimensionality_reduction/ # 降维算法
│   ├── database/              # 数据库管理
│   ├── encoding/              # 特征编码
│   ├── hyperparameter/        # 超参数调优
│   ├── result_generators/     # 结果生成器
│   ├── routes/                # API路由
│   ├── services/              # 业务服务
│   └── utils/                 # 工具函数
├── frontend/                  # 前端代码
│   ├── css/                   # 样式文件
│   ├── js/                    # JavaScript代码
│   │   ├── workflow/          # 工作流相关
│   │   ├── visualization/     # 可视化
│   │   └── utils/             # 工具函数
│   └── assets/                # 静态资源
├── requirements.txt           # Python依赖
├── start.bat                  # Windows启动脚本
└── README.md                  # 项目说明文档
```



## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 支持与反馈

- **问题反馈**: 请在 GitHub Issues 中提交问题
- **功能建议**: 欢迎提出改进建议
- **技术讨论**: 可通过 Issues 进行技术讨论

## 🌟 致谢

感谢所有为项目做出贡献的开发者！

---

**开始你的机器学习之旅吧！无需编程，轻松构建智能模型。**