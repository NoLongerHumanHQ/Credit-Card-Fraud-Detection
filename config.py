"""
Configuration settings for the Credit Card Fraud Detection system.
Contains paths, model parameters, and other constants.
"""
from typing import Dict, List, Any
import os
from pathlib import Path

# Project structure
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "saved_models"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data settings
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
LOCAL_DATA_PATH = DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "Class"

# Feature engineering
FEATURES_TO_SCALE = ["Amount", "Time"]
TIME_FEATURES = ["hour", "day_of_week", "is_weekend"]

# Model parameters
MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "n_jobs": -1
    },
    "random_forest": {
        "n_estimators": 100, 
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "voting": {
        "voting": "soft",
        "weights": [1, 2, 3]  # LR, RF, XGB weights
    }
}

# Model evaluation
CV_FOLDS = 5
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# SMOTE parameters
SMOTE_PARAMS = {
    "sampling_strategy": 0.1,  # Aim for 1:10 ratio instead of 1:1 to avoid extreme oversampling
    "random_state": RANDOM_STATE
}

# Visualization settings
COLORS = {
    "primary": "#007BFF",
    "secondary": "#6C757D",
    "success": "#28A745", 
    "danger": "#DC3545",
    "warning": "#FFC107",
    "info": "#17A2B8"
}

# Risk thresholds for fraud prediction
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.7,
    "high": 0.9
}

# App settings
APP_TITLE = "Credit Card Fraud Detection"
APP_DESCRIPTION = "A lightweight system for detecting fraudulent credit card transactions"
CACHE_TTL = 3600  # Cache time to live in seconds 