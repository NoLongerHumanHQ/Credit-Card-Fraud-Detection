# Credit Card Fraud Detection System

A lightweight, end-to-end machine learning system for detecting fraudulent credit card transactions. This project demonstrates a complete ML pipeline from data processing to model deployment with a focus on performance, interpretability, and clean code structure.

![Fraud Detection Dashboard](https://github.com/Nneji123/Credit-Card-Fraud-Detection/raw/main/image.png)

## 🌟 Features

- **Data Processing Pipeline**: Efficient data handling with automatic optimization and feature engineering
- **Multiple Models**: Logistic Regression (baseline), Random Forest, XGBoost, and Ensemble methods
- **Interactive Dashboard**: Clean UI with model performance metrics, real-time prediction, and batch processing
- **Model Explainability**: SHAP-based interpretability to understand prediction factors
- **Performance Optimization**: Efficient memory usage, model compression, and caching for fast predictions
- **Business Intelligence**: Risk categorization, false positive cost analysis, and A/B testing framework

## 📊 Demo

The application has four main tabs:

1. **Model Performance**: View metrics, ROC curves, and confusion matrices for trained models
2. **Real-time Prediction**: Make predictions on individual transactions with risk assessment
3. **Batch Prediction**: Upload CSV files with multiple transactions for bulk processing
4. **Model Insights**: Analyze feature importance and compare model thresholds

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

The application will automatically download the required dataset on first run.

## 📁 Project Structure

```
credit_card_fraud_detection/
├── app.py                 # Main Streamlit application
├── models/                # Model training and prediction
│   ├── __init__.py
│   ├── model_trainer.py   # Training pipeline
│   ├── fraud_detector.py  # Prediction engine
│   └── saved_models/      # Trained model files
├── data/                  # Data processing
│   ├── __init__.py
│   ├── data_processor.py  # Data preprocessing
│   └── sample_data.csv    # Small sample for demo
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── visualizations.py  # Plotting functions
│   └── metrics.py         # Evaluation metrics
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── config.py              # Configuration settings
```

## 📊 Dataset

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset contains transactions made by credit cards in September 2013 by European cardholders, with 492 frauds out of 284,807 transactions.

Features:
- `Time`: Seconds elapsed between each transaction and the first transaction
- `V1-V28`: Principal components obtained with PCA (anonymized features)
- `Amount`: Transaction amount
- `Class`: 1 for fraudulent transactions, 0 for legitimate ones

The application will automatically download the dataset from a mirror when first run.

## 🎯 Model Performance

Our models achieve the following performance metrics on the test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9734 | 0.8152 | 0.7642 | 0.7889 | 0.9012 |
| Random Forest | 0.9912 | 0.9341 | 0.8372 | 0.8829 | 0.9764 |
| XGBoost | 0.9937 | 0.9624 | 0.8791 | 0.9189 | 0.9853 |
| Voting Classifier | 0.9943 | 0.9582 | 0.8953 | 0.9258 | 0.9891 |

## 🚀 Deployment

The application is optimized for deployment on Streamlit Cloud:

1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the app from your forked repository

Total size is kept under 100MB for quick loading and deployment.

## 📝 API Documentation

The `FraudDetector` class serves as the main prediction engine and can be used independently:

```python
from models.fraud_detector import FraudDetector

# Initialize detector
detector = FraudDetector()

# Load model
detector.load_model()

# Make prediction on a single transaction
transaction = {
    'Time': 43200,  # 12 hours from start
    'Amount': 149.62,
    'V1': -1.359807,
    # ...other V features
}
result = detector.predict(transaction)

# Results include:
# - is_fraud: Boolean indicating fraud detection
# - fraud_probability: Probability of fraud (0-1)
# - risk_level: Categorized risk (low, medium, high, very high)
# - confidence: Confidence of the prediction
# - top_factors: Features that influenced the prediction
```

For batch predictions, use the `predict_batch` method with a list of transactions.

## 🧪 Future Enhancements

- Integration with alerting systems
- API service for real-time fraud checks
- Active learning framework for model retraining
- Time-based feature drift analysis
- User feedback incorporation for false positive reduction

## 🔄 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Dataset provided by the Machine Learning Group at ULB (Université Libre de Bruxelles)
- Inspired by [Nneji123's Credit Card Fraud Detection project](https://github.com/Nneji123/Credit-Card-Fraud-Detection) 