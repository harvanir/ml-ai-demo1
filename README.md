# ML-AI-Demo1

A simple demo for combining Machine Learning and AI in a practical file analysis workflow.

## Overview

This project accepts uploaded Excel or CSV files, summarizes the data, detects anomalies in numeric columns using simple ML/statistics, and optionally uses an LLM to explain the results.

## Quick Start

### Environment Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml-ai-demo1
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and Isolation Forest settings
   ```

3. **Install dependencies:**
   ```bash
   cd python
   pip install -r requirements.txt
   ```

4. **Run the application:**
   - Using scripts (recommended for Windows):
     ```bash
     # Start the application
     scripts\start.bat
     
     # Stop the application
     scripts\stop.bat
     ```
   - Manual:
     ```bash
     cd python
     python -m app.main
     ```

5. **Test the API:**
   ```bash
   curl -X POST "http://localhost:8000/analyze" -F "file=@../sample_data/sample.csv"
   ```

### Configuration Options
- **OpenAI API Key**: Required for AI explanations (set in `.env`)
- **Isolation Forest**: Configurable parameters for anomaly detection (see [Isolation Forest Configuration Guide](docs/isolation_forest_config.md))

## 🤖 AI Assistant Guidelines

This project includes guidelines for AI assistants to ensure consistent development practices. See [.ai-guidelines.md](.ai-guidelines.md) for:
- Development workflow checklists
- Testing requirements before making changes
- Code quality standards
- Task-specific instructions

## Machine Learning Use Cases

### 🎯 **Primary Use Case: Anomaly Detection in Data Files**

**Business Problems Solved:**
- **Data Quality Monitoring**: Automatically detect unusual values in datasets
- **Fraud Detection**: Identify suspicious transactions or measurements
- **Quality Control**: Flag products with out-of-spec values
- **IoT Monitoring**: Detect abnormal sensor readings
- **Financial Analysis**: Spot unusual financial transactions

**Real-World Applications:**
- E-commerce: Detect fraudulent orders with abnormal pricing
- Manufacturing: Identify defective products with unusual measurements
- Finance: Flag suspicious transactions with extreme values
- Healthcare: Monitor patient data for abnormal vital signs
- IoT: Detect malfunctioning sensors with outlier readings

### 🧠 **ML Learning Patterns Demonstration**

This project now includes comprehensive demonstrations of **6 major machine learning learning patterns**:

#### **1. Supervised Learning** 🎓
- **Classification**: Predict categories (e.g., spam/not-spam, fraud/legitimate)
- **Regression**: Predict continuous values (e.g., price prediction, demand forecasting)
- **Algorithms**: Random Forest, Decision Trees, Linear Models

#### **2. Unsupervised Learning** 🔍
- **Clustering**: Group similar data points (e.g., customer segmentation)
- **Anomaly Detection**: Find outliers and unusual patterns
- **Algorithms**: K-means, DBSCAN, Isolation Forest

#### **3. Semi-Supervised Learning** 📚
- **Partial Labels**: Learn from datasets with some labeled and mostly unlabeled data
- **Label Propagation**: Spread labels from known to unknown examples
- **Use Cases**: Medical diagnosis with limited expert labels

#### **4. Reinforcement Learning** 🎮
- **Agent Learning**: Learn through trial-and-error with rewards/penalties
- **Decision Making**: Optimize actions in dynamic environments
- **Example**: Q-learning agent for simple decision problems

#### **5. Deep Learning** 🧠
- **Neural Networks**: Multi-layer architectures for complex pattern recognition
- **Automatic Feature Learning**: No manual feature engineering required
- **Frameworks**: PyTorch for classification and regression

#### **6. Ensemble Learning** 👥
- **Model Combination**: Combine multiple models for better performance
- **Voting & Boosting**: Hard/soft voting, bagging, gradient boosting
- **Improved Accuracy**: Often outperforms individual models
## Current Status ✅

**All 6 ML Learning Patterns Successfully Implemented and Tested:**

### **✅ Supervised Learning**
- Random Forest classification and regression
- Automatic task detection (classification vs regression)
- Performance metrics and feature importance

### **✅ Unsupervised Learning**
- K-means clustering with optimal k selection
- Anomaly detection using Isolation Forest
- Cluster quality evaluation

### **✅ Semi-Supervised Learning**
- Label propagation algorithm
- Confidence scoring for unlabeled data
- Mixed labeled/unlabeled data handling

### **✅ Reinforcement Learning**
- Q-learning agent implementation
- Grid world environment simulation
- Learning curve analysis

### **✅ Deep Learning**
- Multi-layer perceptron (MLP) with scikit-learn
- Automatic architecture configuration
- Early stopping and convergence monitoring

### **✅ Ensemble Learning**
- Voting classifiers and regressors
- Multiple base models (LR, SVM, DT, RF, GB)
- Performance comparison with individual models

## Quick Start

1. **Install dependencies:**
   ```bash
   cd python
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python -m app.main
   ```

3. **Test all patterns:**
   ```bash
   python -m python.tests.test_ml_patterns
   ```

4. **Interactive API:** Visit `http://localhost:8000/docs`

## Project Structure 📁

```
✅ MODULAR ARCHITECTURE
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings with modern Pydantic
│   ├── api/routes.py        # REST API endpoints
│   ├── models/response.py   # Pydantic data models
│   ├── services/            # Core business logic services
│   │   ├── __init__.py      # Package initialization
│   │   ├── anomaly_detector.py # IQR anomaly detection (CORE FEATURE)
│   │   ├── ai_explainer.py  # AI summary generation
│   │   └── ml_patterns/     # 6 ML learning patterns
│   └── utils/               # Utility functions
│       ├── __init__.py      # Package initialization
│       ├── file_reader.py   # CSV/Excel file I/O utilities
│       └── dataframe_helper.py # Data processing utilities
├── tests/                   # Unit tests
└── requirements.txt         # All dependencies pinned
```

## Code Quality & Organization ✅

- **Modular Architecture**: Clean separation of concerns with services, models, and utilities
- **Type Hints**: Full type annotations for better code maintainability  
- **Error Handling**: Comprehensive error handling in all ML pattern implementations
- **Documentation**: Detailed docstrings and API documentation
- **Testing**: Unit tests for core functionality
- **Dependencies**: Properly pinned requirements with all necessary packages
## Technical Implementation

### 🤖 **ML Algorithms Used**

#### **Anomaly Detection Methods**
1. **IQR (Interquartile Range) Method** - Primary method for small datasets
   - Calculates Q1, Q3, and IQR
   - Flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   - Robust for small datasets and various distributions

2. **Z-Score Method** - Statistical method for normally distributed data
   - Calculates standard deviations from mean
   - Flags values with |z-score| > threshold
   - Good for datasets with known normal distribution

3. **Isolation Forest Method** - Advanced method for large datasets (100k+)
   - Builds ensemble of isolation trees
   - Anomalies have shorter path lengths in trees
   - Highly scalable O(n log n) complexity
   - Excellent for high-dimensional data

#### **Isolation Forest Configuration**

See [Isolation Forest Configuration Guide](docs/isolation_forest_config.md) for detailed setup and usage instructions.

#### **ML Learning Patterns Implementation**

**Supervised Learning:**
- Random Forest for both classification and regression
- Automatic task detection based on target variable distribution
- Feature importance analysis

**Unsupervised Learning:**
- K-means clustering with elbow method for optimal k
- Anomaly detection using isolation principles
- Silhouette analysis for cluster quality

**Semi-Supervised Learning:**
- Label propagation algorithm
- Handles partially labeled datasets
- Confidence scoring for propagated labels

**Reinforcement Learning:**
- Q-learning agent implementation
- Simple grid world environment
- Reward-based learning demonstration

**Deep Learning:**
- PyTorch neural networks
- Automatic architecture selection (classification vs regression)
- Dropout regularization and early stopping

**Ensemble Learning:**
- Multiple base models (Logistic Regression, SVM, Decision Trees, etc.)
- Voting classifiers/regressors
- Performance comparison with individual models

#### **Data Processing Pipeline**
```
File Upload → Data Reading → Type Detection → Summary Generation → Anomaly Detection → AI Explanation → JSON Response
```

**ML Patterns Pipeline:**
```
File Upload → Pattern Selection → Data Preprocessing → Model Training → Performance Evaluation → Pattern-Specific Insights → JSON Response
```

### 📊 **Technical Features**

- **File Support**: CSV and Excel (.xlsx, .xls) files
- **Data Types**: Automatic detection of numeric columns
- **Statistics**: Total rows, columns, numeric column identification
- **Anomaly Scoring**: Outlier detection with confidence metrics
- **AI Integration**: Optional LLM-powered explanations (requires OpenAI API key)

## API Documentation

### **Endpoints**

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `GET /ml-patterns`
Get available ML learning patterns.

**Response:**
```json
{
  "patterns": ["supervised", "unsupervised", "semi_supervised", "reinforcement", "deep_learning", "ensemble"]
}
```

#### `POST /analyze`
Analyze uploaded file for data summary and anomalies.

**Parameters:**
- `file` (required): CSV or Excel file
- `use_ai` (optional): Boolean flag for AI explanation (default: false)

**Example Request:**
```bash
curl -X POST -F "file=@sample_data/sample.csv" http://localhost:8000/analyze
```

**Response:**
```json
{
  "summary": {
    "total_rows": 100,
    "total_columns": 5,
    "numeric_columns": ["price", "quantity", "rating"]
  },
  "anomalies": [
    {
      "column": "price",
      "value": 999.99,
      "index": 42,
      "is_outlier": true
    }
  ],
  "ai_summary": "Detected an unusually high price value that may indicate data entry error or special pricing."
}
```

#### `POST /analyze-pattern`
Analyze uploaded file using specified ML learning pattern.

**Parameters:**
- `file` (required): CSV or Excel file
- `pattern` (optional): ML pattern to use (default: "supervised")

**Available Patterns:**
- `supervised` - Classification and regression
- `unsupervised` - Clustering and anomaly detection
- `semi_supervised` - Learning with partial labels
- `reinforcement` - Q-learning agent demonstration
- `deep_learning` - Neural network classification/regression
- `ensemble` - Combined model performance

**Example Request:**
```bash
curl -X POST -F "file=@sample_data/sample.csv" -F "pattern=supervised" http://localhost:8000/analyze-pattern
```

**Response:**
```json
{
  "summary": {
    "total_rows": 100,
    "total_columns": 5,
    "numeric_columns": ["price", "quantity", "rating"]
  },
  "ml_pattern": {
    "pattern": "supervised",
    "type": "classification",
    "model_performance": {
      "accuracy": 0.85,
      "f1_score": 0.83
    },
    "insights": [
      "🎯 Supervised learning model trained successfully",
      "📊 Classification task with 3 classes",
      "🎯 Model accuracy: 85.0%"
    ]
  },
  "pattern_used": "supervised"
}
```

## Structure

- `python/` - Python implementation (FastAPI)
- `golang/` - Future Golang implementation
- `docs/` - Documentation
- `shared/` - Shared schemas and examples
- `sample_data/` - Sample files for testing

## Getting Started

### Python

1. Install dependencies:
   ```bash
   cd python
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python -m app.main
   ```

3. Test the API:
   ```bash
   # From project root
   python -m python.tests.test_ml_demo
   ```

## Sample Data

The `sample_data/sample.csv` contains test data with intentional anomalies:
- Normal products with typical pricing
- One product with extreme price (999.99) - detected as anomaly
- Products with varying quantities - some flagged as outliers

## Development

### Running Tests
```bash
cd python
python -m pytest tests/ -v
```

### API Documentation
When running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Future Enhancements

- Multiple anomaly detection algorithms
- Time-series analysis
- Multi-sheet Excel support
- Advanced AI explanations
- Batch processing
- Real-time monitoring dashboard