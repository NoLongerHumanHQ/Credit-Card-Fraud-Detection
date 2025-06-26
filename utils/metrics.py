"""
Utility functions for model evaluation and performance metrics.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a model's performance on test data.
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: Test labels
        threshold: Threshold for binary classification
        
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba)
    }
    
    logger.info(f"Model evaluation metrics: {metrics}")
    
    return metrics

def get_confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        np.ndarray: Confusion matrix as a numpy array
    """
    return confusion_matrix(y_true, y_pred)

def get_classification_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    as_dict: bool = False
) -> Union[str, Dict]:
    """
    Generate a classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        as_dict: Whether to return the report as a dictionary
        
    Returns:
        Union[str, Dict]: Classification report as string or dictionary
    """
    return classification_report(y_true, y_pred, output_dict=as_dict)

def find_optimal_threshold(
    y_true: Union[List, np.ndarray, pd.Series],
    y_proba: Union[List, np.ndarray, pd.Series],
    metric: str = "f1"
) -> float:
    """
    Find the optimal threshold for binary classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for the positive class
        metric: Metric to optimize ('f1', 'precision', 'recall', or 'balanced')
        
    Returns:
        float: Optimal threshold
    """
    # Get precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Add endpoint
    thresholds = np.append(thresholds, 1.0)
    
    if metric == "f1":
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        return thresholds[np.argmax(f1_scores)]
    
    elif metric == "precision":
        return thresholds[np.argmax(precision)]
    
    elif metric == "recall":
        # For recall, we want the threshold that gives at least 0.8 recall
        # with the highest possible precision
        valid_indices = recall >= 0.8
        if any(valid_indices):
            max_precision_idx = np.argmax(precision[valid_indices])
            return thresholds[valid_indices][max_precision_idx]
        return thresholds[np.argmax(recall)]  # Default if no threshold meets recall requirement
    
    elif metric == "balanced":
        # Balance precision and recall
        balance = precision + recall - np.abs(precision - recall)
        return thresholds[np.argmax(balance)]
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def get_threshold_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_proba: Union[List, np.ndarray, pd.Series],
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Calculate metrics for different threshold values.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for the positive class
        thresholds: List of thresholds to evaluate
        
    Returns:
        pd.DataFrame: DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (np.array(y_proba) >= threshold).astype(int)
        
        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        })
    
    return pd.DataFrame(results)

def calculate_cost_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    avg_transaction_amount: float = 100.0,
    cost_false_positive: float = 10.0,
    cost_false_negative: float = None
) -> Dict[str, float]:
    """
    Calculate business cost metrics for fraud detection.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        avg_transaction_amount: Average transaction amount
        cost_false_positive: Cost of investigating a false positive
        cost_false_negative: Cost of not catching fraud (if None, uses avg_transaction_amount)
        
    Returns:
        Dict[str, float]: Dictionary with cost metrics
    """
    if cost_false_negative is None:
        cost_false_negative = avg_transaction_amount
    
    # Calculate confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate costs
    total_cost_false_positives = fp * cost_false_positive
    total_cost_false_negatives = fn * cost_false_negative
    total_cost = total_cost_false_positives + total_cost_false_negatives
    
    # Calculate savings
    total_fraud_amount = sum(y_true) * avg_transaction_amount
    saved_amount = tp * avg_transaction_amount
    
    return {
        "total_cost": total_cost,
        "false_positive_cost": total_cost_false_positives,
        "false_negative_cost": total_cost_false_negatives,
        "saved_amount": saved_amount,
        "total_fraud_amount": total_fraud_amount,
        "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0
    }

def get_risk_levels(
    y_proba: Union[List, np.ndarray, pd.Series]
) -> List[str]:
    """
    Categorize transactions into risk levels based on fraud probability.
    
    Args:
        y_proba: Predicted probabilities for the positive class
        
    Returns:
        List[str]: Risk levels ('low', 'medium', 'high')
    """
    risk_levels = []
    
    for prob in y_proba:
        if prob < config.RISK_THRESHOLDS["low"]:
            risk_levels.append("low")
        elif prob < config.RISK_THRESHOLDS["medium"]:
            risk_levels.append("medium")
        elif prob < config.RISK_THRESHOLDS["high"]:
            risk_levels.append("high")
        else:
            risk_levels.append("very high")
    
    return risk_levels 