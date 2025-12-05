# ML Visual Builder - Machine Learning Visual Platform

## 🚀 Project Overview

ML Visual Builder is a web-based machine learning visual platform that allows users to build, train, and deploy machine learning models without writing code through a drag-and-drop interface. The platform integrates complete machine learning workflows including data preprocessing, feature engineering, model training, hyperparameter tuning, and result visualization.

## ✨ Core Features

### 📊 Data Management
- **Multi-format Support**: Import common data formats like CSV, Excel, JSON
- **Data Preview**: Real-time viewing of data structure and statistical information
- **Data Cleaning**: Missing value handling, outlier detection, data standardization
- **Feature Engineering**: Feature selection, feature transformation, feature encoding

### 🧩 Visual Workflow
- **Drag-and-Drop Construction**: Build machine learning workflows by dragging components onto canvas
- **Node-based Design**: Data nodes, preprocessing nodes, algorithm nodes, evaluation nodes
- **Process Visualization**: Real-time display of data processing and model training workflows
- **Workflow Management**: Save, load, and share workflow configurations

### 🤖 Algorithm Library

#### Classification Algorithms
- **Logistic Regression** - Classic linear classification model
- **Decision Tree Classifier** - Tree-based classification algorithm
- **Random Forest Classifier** - Ensemble learning with multiple decision trees
- **SVM Classifier** - Classification algorithm based on maximum margin
- **KNN Classifier** - Distance-based classification algorithm
- **Naive Bayes Classifier** - Probabilistic classification based on Bayes theorem
- **Gradient Boosting Classifier** - Ensemble learning with sequential optimization
- **MLP Classifier** - Neural network classifier
- **XGBoost Classifier** - Optimized gradient boosting algorithm
- **LightGBM Classifier** - Gradient boosting framework developed by Microsoft
- **CatBoost Classifier** - Gradient boosting algorithm for categorical features

#### Regression Algorithms
- **Linear Regression** - Basic linear regression model
- **Ridge Regression** - L2-regularized linear regression
- **Lasso Regression** - L1-regularized linear regression
- **ElasticNet Regression** - L1+L2 regularized linear regression
- **Decision Tree Regressor** - Tree-based regression algorithm
- **Random Forest Regressor** - Ensemble learning regression algorithm
- **Gradient Boosting Regressor** - Ensemble learning regression algorithm
- **Support Vector Regressor (SVR)** - SVM-based regression algorithm
- **MLP Regressor** - Neural network regressor
- **XGBoost Regressor** - Optimized gradient boosting regression
- **LightGBM Regressor** - Microsoft's gradient boosting regression

#### Clustering Algorithms
- **K-Means Clustering** - Classic distance-based clustering algorithm
- **DBSCAN Clustering** - Density-based clustering algorithm
- **Gaussian Mixture Model (GMM)** - Probability distribution-based clustering
- **Hierarchical Clustering** - Builds clustering hierarchy
- **Spectral Clustering** - Graph theory-based clustering algorithm

#### Dimensionality Reduction Algorithms
- **Principal Component Analysis (PCA)** - Linear dimensionality reduction maximizing variance
- **Linear Discriminant Analysis (LDA)** - Supervised linear dimensionality reduction
- **t-SNE** - Non-linear dimensionality reduction preserving local structure
- **UMAP** - Non-linear dimensionality reduction preserving global and local structure

### ⚙️ Model Training & Tuning
- **Automatic Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Cross-Validation**: K-fold cross-validation, leave-one-out validation
- **Model Evaluation**: Accuracy, precision, recall, F1-score, AUC, RMSE, MAE, and other metrics
- **Model Persistence**: Trained models can be saved as files or stored in database

### 📈 Result Visualization
- **Training Process Visualization**: Loss curves, accuracy curves, feature importance
- **Prediction Result Visualization**: Confusion matrices, ROC curves, residual plots, clustering results
- **Interactive Charts**: Support zooming, panning, data point inspection
- **Report Generation**: Automatic generation of model evaluation reports

## 🎯 Project Highlights

### 🎨 User-Friendly Interface
- **Zero-Code Operation**: Complete machine learning tasks through visual interface without programming experience
- **Intuitive Design**: Clean and clear interface design reduces learning curve
- **Real-time Feedback**: Real-time display of processing results and status during operations

## 🚀 Quick Start

### Environment Requirements
- Python 3.10 or higher
- Windows/macOS/Linux operating system
- 4GB+ RAM (8GB recommended)
- 2GB+ available disk space

### Installation Steps

#### Method 1: Using Startup Script (Recommended)
1. Clone or download the project locally
2. Double-click `start.bat` (Windows) or execute `./start.sh` (Linux/macOS)
3. The script will automatically:
   - Check Python environment
   - Install uv package manager
   - Create virtual environment
   - Install all dependencies
   - Initialize database
   - Start Flask server

#### Method 2: Manual Installation
```bash
# 1. Clone the project
git clone https://github.com/673279998/ML_Visual_Builder.git
cd ML_Visual_Builder

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize database
python backend/database/models.py

# 6. Start server
cd backend
python app.py
```

### Running the Platform
1. After successful startup, access in browser: `http://localhost:5000`
2. The platform main interface includes:
   - **Workflow**: Drag-and-drop machine learning workflow construction
   - **Data Management**: Import, view, process data
   - **Model Management**: View trained models
   - **Model Prediction**: Use trained models for prediction

## 📖 User Guide

### 1. Data Import
- Click "Data Management" page
- Select "Import Data" button
- Upload CSV, Excel, or JSON files
- Preview data and confirm import

### 2. Create Workflow
- Go to "Workflow" page
- Drag components from left toolbar to canvas:
  - **Data Source**: Select imported datasets
  - **Preprocessing**: Data cleaning, feature engineering
  - **Algorithm**: Select machine learning algorithms
  - **Evaluation**: Model evaluation metrics
- Connect components to build complete workflow

### 3. Model Training
- Configure algorithm parameters
- Set train/test split ratio
- Click "Execute" button to start training
- View training progress and results in real-time

### 4. Result Analysis
- View model evaluation metrics
- Analyze visualization charts
- Save trained models
- Export evaluation reports

## 🏗️ Project Structure

```
ML_Visual_Builder/
├── backend/                    # Backend code
│   ├── algorithms/            # Algorithm implementations
│   │   ├── classification/    # Classification algorithms
│   │   ├── regression/        # Regression algorithms
│   │   ├── clustering/        # Clustering algorithms
│   │   └── dimensionality_reduction/ # Dimensionality reduction algorithms
│   ├── database/              # Database management
│   ├── encoding/              # Feature encoding
│   ├── hyperparameter/        # Hyperparameter tuning
│   ├── result_generators/     # Result generators
│   ├── routes/                # API routes
│   ├── services/              # Business services
│   └── utils/                 # Utility functions
├── frontend/                  # Frontend code
│   ├── css/                   # Style files
│   ├── js/                    # JavaScript code
│   │   ├── workflow/          # Workflow related
│   │   ├── visualization/     # Visualization
│   │   └── utils/             # Utility functions
│   └── assets/                # Static resources
├── requirements.txt           # Python dependencies
├── start.bat                  # Windows startup script
├── README.md                  # English documentation
└── README_CN.md               # Chinese documentation
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📞 Support & Feedback

- **Issue Reporting**: Please submit issues on GitHub Issues
- **Feature Suggestions**: Welcome to suggest improvements
- **Technical Discussion**: Technical discussions can be conducted through Issues

## 🌟 Acknowledgments

Thanks to all developers who contributed to the project!

---

**Start your machine learning journey! No coding required, build intelligent models easily.**