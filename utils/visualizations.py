"""
Visualization utilities for credit card fraud detection.
Contains functions for creating various plots and charts.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sys
import logging

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    normalize: bool = False
) -> go.Figure:
    """
    Plot confusion matrix using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        go.Figure: Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # Get confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    
    # Create confusion matrix plot
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        text=[[f"{tn:.2%}" if normalize else f"{tn}", f"{fp:.2%}" if normalize else f"{fp}"],
              [f"{fn:.2%}" if normalize else f"{fn}", f"{tp:.2%}" if normalize else f"{tp}"]],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        xaxis=dict(side='top'),
        height=500,
        width=600,
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    return fig

def plot_roc_curve(
    y_true: Union[List, np.ndarray, pd.Series],
    y_proba: Union[List, np.ndarray, pd.Series]
) -> go.Figure:
    """
    Plot ROC curve using Plotly.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        go.Figure: Plotly figure object
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color=config.COLORS['primary'], width=2)
    ))
    
    # Add diagonal reference line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.5)',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        width=700,
        height=500,
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.8)')
    )
    
    return fig

def plot_precision_recall_curve(
    y_true: Union[List, np.ndarray, pd.Series],
    y_proba: Union[List, np.ndarray, pd.Series]
) -> go.Figure:
    """
    Plot Precision-Recall curve using Plotly.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        go.Figure: Plotly figure object
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    
    # Add Precision-Recall curve
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.3f})',
        line=dict(color=config.COLORS['primary'], width=2)
    ))
    
    # Add reference line (random classifier)
    baseline = np.sum(y_true) / len(y_true)  # Proportion of positive class
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[baseline, baseline],
        mode='lines',
        name=f'Random Classifier (Precision = {baseline:.3f})',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        width=700,
        height=500,
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.8)')
    )
    
    return fig

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> go.Figure:
    """
    Plot feature importance for a tree-based model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        top_n: Number of top features to display
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute.")
        
        # If model has coef_ attribute (e.g., logistic regression), use that instead
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.error("Model has neither feature_importances_ nor coef_ attribute.")
            return None
    else:
        importances = model.feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance and get top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        importance_df,
        y='feature',
        x='importance',
        orientation='h',
        color='importance',
        color_continuous_scale='Blues',
        title=f'Top {top_n} Feature Importances'
    )
    
    fig.update_layout(
        yaxis=dict(title=''),
        xaxis=dict(title='Importance'),
        height=30 * min(top_n, len(feature_names)) + 100,  # Dynamic height based on number of features
        coloraxis_showscale=False
    )
    
    return fig

def plot_threshold_metrics(
    df_metrics: pd.DataFrame
) -> go.Figure:
    """
    Plot metrics vs threshold values.
    
    Args:
        df_metrics: DataFrame with columns 'threshold', 'precision', 'recall', 'f1', 'accuracy'
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    colors = [config.COLORS['danger'], config.COLORS['success'], config.COLORS['primary'], config.COLORS['secondary']]
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Scatter(
            x=df_metrics['threshold'],
            y=df_metrics[metric],
            mode='lines+markers',
            name=metric.capitalize(),
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title='Model Performance Metrics vs Threshold',
        xaxis=dict(title='Threshold', tickformat='.1f'),
        yaxis=dict(title='Score', range=[0, 1.05]),
        width=700,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 20,
    plot_type: str = 'bar'
) -> Any:
    """
    Create SHAP summary plot for model interpretability.
    
    Args:
        model: Trained model
        X: Feature matrix
        max_display: Maximum number of features to display
        plot_type: Type of plot ('bar', 'dot', 'violin')
        
    Returns:
        matplotlib.figure.Figure: SHAP summary plot
    """
    # For XGBoost models
    if hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model)
    # For scikit-learn models
    else:
        explainer = shap.Explainer(model, X)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # Convert to expected format if needed
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # For binary classification, get positive class values
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    if plot_type == 'bar':
        shap.summary_plot(shap_values, X, plot_type='bar', max_display=max_display, show=False)
    elif plot_type == 'violin':
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    else:
        shap.summary_plot(shap_values, X, plot_type='dot', max_display=max_display, show=False)
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_shap_waterfall(
    model: Any,
    X: pd.DataFrame,
    instance_index: int = 0
) -> Any:
    """
    Create SHAP waterfall plot for a single instance.
    
    Args:
        model: Trained model
        X: Feature matrix
        instance_index: Index of the instance to explain
        
    Returns:
        matplotlib.figure.Figure: SHAP waterfall plot
    """
    # For XGBoost models
    if hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model)
    # For scikit-learn models
    else:
        explainer = shap.Explainer(model, X)
    
    # Get single instance
    instance = X.iloc[instance_index:instance_index+1]
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(instance)
    
    # Convert to expected format if needed
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # For binary classification, get positive class values
    
    # Create waterfall plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explainer.expected_value if not isinstance(explainer.expected_value, list) 
                        else explainer.expected_value[1], 
                        shap_values[0], 
                        feature_names=X.columns, 
                        max_display=15,
                        show=False)
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_model_comparison(
    models_metrics: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Create comparison plot for multiple models.
    
    Args:
        models_metrics: Dictionary with model names as keys and metrics dictionaries as values
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create metrics dataframe
    df = pd.DataFrame(columns=['model', 'metric', 'value'])
    
    for model_name, metrics in models_metrics.items():
        for metric_name, value in metrics.items():
            df = df.append({
                'model': model_name,
                'metric': metric_name,
                'value': value
            }, ignore_index=True)
    
    # Create grouped bar chart
    fig = px.bar(
        df,
        x='model',
        y='value',
        color='metric',
        barmode='group',
        color_discrete_sequence=[config.COLORS['primary'], config.COLORS['success'], 
                                config.COLORS['danger'], config.COLORS['warning'],
                                config.COLORS['info']],
        title='Model Comparison',
        labels={'value': 'Score', 'model': 'Model', 'metric': 'Metric'}
    )
    
    fig.update_layout(
        yaxis=dict(title='Score', range=[0, 1.05]),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=500,
        width=700
    )
    
    return fig

def plot_transaction_risk(risk_score: float) -> go.Figure:
    """
    Create a gauge chart showing transaction risk level.
    
    Args:
        risk_score: Risk score between 0 and 1
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Define color zones
    if risk_score < config.RISK_THRESHOLDS['low']:
        color = config.COLORS['success']
        risk_level = 'Low'
    elif risk_score < config.RISK_THRESHOLDS['medium']:
        color = config.COLORS['warning']
        risk_level = 'Medium'
    elif risk_score < config.RISK_THRESHOLDS['high']:
        color = config.COLORS['danger']
        risk_level = 'High'
    else:
        color = 'darkred'
        risk_level = 'Very High'
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Fraud Risk Level: {risk_level}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, config.RISK_THRESHOLDS['low']], 'color': 'rgba(0, 250, 0, 0.3)'},
                {'range': [config.RISK_THRESHOLDS['low'], config.RISK_THRESHOLDS['medium']], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [config.RISK_THRESHOLDS['medium'], config.RISK_THRESHOLDS['high']], 'color': 'rgba(255, 150, 0, 0.3)'},
                {'range': [config.RISK_THRESHOLDS['high'], 1], 'color': 'rgba(250, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        },
        number={'valueformat': '.3f'}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig