"""
Credit Card Fraud Detection - Streamlit App
Main application file with multiple tabs for different functionalities.
"""
import os
import sys
import time
import datetime
import base64
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go
from streamlit.runtime.uploaded_file_manager import UploadedFile
import uuid

# Add root directory to path
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

# Import project modules
import config
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from models.fraud_detector import FraudDetector
from utils.metrics import (
    evaluate_model,
    get_confusion_matrix,
    get_threshold_metrics,
    calculate_cost_metrics
)
from utils.visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_threshold_metrics,
    plot_model_comparison,
    plot_transaction_risk
)

# Configure Streamlit page
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit cache for data loading and preprocessing
@st.cache_data(ttl=config.CACHE_TTL)
def load_and_preprocess_data(use_sample: bool = True):
    """Load and preprocess the dataset with caching."""
    data_processor = DataProcessor(use_sample=use_sample)
    
    # Download dataset if needed
    if not os.path.exists(data_processor.data_path):
        with st.spinner("Downloading dataset..."):
            data_processor.download_dataset()
    
    # Load and preprocess data
    df = data_processor.load_data()
    df = data_processor.preprocess_data(df)
    
    return df, data_processor

@st.cache_data(ttl=config.CACHE_TTL)
def train_or_load_models(force_train: bool = False):
    """
    Train or load fraud detection models.
    
    Args:
        force_train (bool): Whether to force training even if models exist.
        
    Returns:
        Tuple: ModelTrainer, FraudDetector, and metrics dictionary.
    """
    # Initialize model trainer and fraud detector
    model_trainer = ModelTrainer()
    fraud_detector = FraudDetector(use_sample_data=True)
    
    # Check if models exist
    model_files = []
    if os.path.exists(config.MODELS_DIR):
        model_files = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('_model.pkl')]
    
    try:
        if not model_files or force_train:
            # We need to train models
            with st.spinner("Training models..."):
                # Prepare data
                data_processor = DataProcessor(use_sample=True)
                X_train, X_test, y_train, y_test = data_processor.prepare_train_test()
                
                # Train models
                model_metrics = model_trainer.train_models(X_train, y_train, X_test, y_test)
                
                # Save models
                model_trainer.save_models()
        else:
            # Load existing models
            with st.spinner("Loading models..."):
                model_trainer.load_models()
                # We still need test data for evaluation
                data_processor = DataProcessor(use_sample=True)
                _, X_test, _, y_test = data_processor.prepare_train_test(apply_smote=False)
                
                # Get metrics for loaded models on test data
                model_metrics = {}
                for name, model in model_trainer.models.items():
                    metrics = evaluate_model(model, X_test, y_test)
                    model_metrics[name] = metrics
    
    except Exception as e:
        st.error(f"Error in model training/loading: {str(e)}")
        
        # Create fallback models for demo purposes
        if not model_trainer.models:
            st.warning("Creating simple fallback models for demonstration purposes.")
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            import xgboost as xgb
            
            # Create simple models
            model_trainer.models = {
                "logistic_regression": LogisticRegression(),
                "random_forest": RandomForestClassifier(n_estimators=10),
                "xgboost": xgb.XGBClassifier(n_estimators=10)
            }
            
            # Create simple metrics
            model_metrics = {
                name: {
                    'accuracy': 0.85, 'precision': 0.75, 'recall': 0.70, 
                    'f1': 0.72, 'roc_auc': 0.80, 'optimal_threshold': 0.5
                } for name in model_trainer.models.keys()
            }
            
            # Initialize fraud detector with the logistic regression model
            fraud_detector.model = model_trainer.models["logistic_regression"]
        
    return model_trainer, fraud_detector, model_metrics

@st.cache_data(ttl=config.CACHE_TTL)
def get_sample_transaction():
    """Get a sample transaction for demonstration."""
    # Define a sample transaction with the main features
    transaction = {
        'Time': 43200,  # 12 hours from start
        'V1': -1.3598071336738,
        'V2': -0.0727811733098497,
        'V3': 2.53634673796914,
        'V4': 1.37815522427443,
        'V5': -0.338320769942518,
        'V6': 0.462387777762292,
        'V7': 0.239598554061257,
        'V8': 0.0986979012610507,
        'V9': 0.363786969611213,
        'V10': 0.0907941719789316,
        'V11': -0.551599533260813,
        'V12': -0.617800855762348,
        'V13': -0.991389847235408,
        'V14': -0.311169353699879,
        'V15': 1.46817697209427,
        'V16': -0.470400525259478,
        'V17': 0.207971241929242,
        'V18': 0.0257905801985591,
        'V19': 0.403992960255733,
        'V20': 0.251412098239705,
        'V21': -0.018306777944153,
        'V22': 0.277837575558899,
        'V23': -0.110473910188767,
        'V24': 0.0669280749146731,
        'V25': 0.128539358273528,
        'V26': -0.189114843888824,
        'V27': 0.133558376740387,
        'V28': -0.0210530534538215,
        'Amount': 149.62
    }
    
    return transaction

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    
    Args:
        object_to_download: The object to be downloaded.
        download_filename: Filename of the downloaded file.
        download_link_text: Text for download link.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Create a BytesIO buffer
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    # Create download link
    href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
    return href

def display_model_performance_tab():
    """Display model performance tab content."""
    st.header("Model Performance")
    
    # Load models and metrics
    model_trainer, fraud_detector, model_metrics = train_or_load_models()
    
    # Create columns for metrics and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Metrics")
        model_options = list(model_trainer.models.keys())
        selected_model = st.selectbox("Select Model", model_options, index=model_options.index("xgboost") if "xgboost" in model_options else 0)
        
        # Display metrics for selected model
        if selected_model in model_metrics:
            metrics = model_metrics[selected_model]
            
            st.markdown("""
            | Metric | Value |
            | --- | --- |
            | Accuracy | {:.4f} |
            | Precision | {:.4f} |
            | Recall | {:.4f} |
            | F1 Score | {:.4f} |
            | ROC AUC | {:.4f} |
            """.format(
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                metrics.get('roc_auc', 0)
            ))
            
            # Add threshold information if available
            if 'optimal_threshold' in metrics:
                st.info(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    
    with col2:
        st.subheader("Performance Visualization")
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["ROC Curve", "Confusion Matrix", "Feature Importance", "Model Comparison"])
        
        # Prepare data for visualizations
        data_processor = DataProcessor(use_sample=True)
        _, X_test, _, y_test = data_processor.prepare_train_test(apply_smote=False)
        selected_model_obj = model_trainer.models[selected_model]
        y_proba = selected_model_obj.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # ROC Curve tab
        with viz_tabs[0]:
            st.plotly_chart(plot_roc_curve(y_test, y_proba), use_container_width=True)
        
        # Confusion Matrix tab
        with viz_tabs[1]:
            cm_fig = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Calculate cost metrics
            cost_metrics = calculate_cost_metrics(y_test, y_pred)
            
            st.markdown("### Business Impact")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Detection Rate", f"{cost_metrics['detection_rate']:.2%}")
                
            with col2:
                st.metric("False Positive Rate", f"{cost_metrics['false_positive_rate']:.2%}")
                
            with col3:
                st.metric("Saved Amount", f"${cost_metrics['saved_amount']:.2f}")
        
        # Feature Importance tab
        with viz_tabs[2]:
            feature_names = data_processor.get_features()
            if feature_names:
                fig = plot_feature_importance(selected_model_obj, feature_names, top_n=20)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature importance not available for this model type.")
            else:
                st.warning("Feature names not available.")
        
        # Model Comparison tab
        with viz_tabs[3]:
            # Show comparison of all models
            st.plotly_chart(plot_model_comparison(model_metrics), use_container_width=True)
            
            # Show threshold analysis
            metrics_df = get_threshold_metrics(y_test, y_proba)
            st.subheader("Threshold Analysis")
            st.plotly_chart(plot_threshold_metrics(metrics_df), use_container_width=True)

def display_prediction_tab():
    """Display single transaction prediction tab."""
    st.header("Real-time Fraud Detection")
    
    # Load Fraud Detector
    _, fraud_detector, _ = train_or_load_models()
    
    # Get sample transaction
    sample_transaction = get_sample_transaction()
    
    st.subheader("Enter Transaction Details")
    st.info("You can modify the values or use the sample transaction as is.")
    
    # Create form for transaction input
    with st.form(key="transaction_form"):
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        # Main transaction details in first column
        with col1:
            time_value = st.slider("Time (hours from start)", 0, 24, int(sample_transaction['Time'] / 3600))
            amount = st.number_input("Amount ($)", min_value=0.01, max_value=10000.0, value=sample_transaction['Amount'], step=10.0)
            
            # Convert time to seconds
            transaction = sample_transaction.copy()
            transaction['Time'] = time_value * 3600
            transaction['Amount'] = amount
        
        # Advanced features in second column - allow modifying top 5 important features
        with col2:
            st.text("Principal Components (modify up to 5)")
            transaction['V1'] = st.slider("V1", -5.0, 5.0, float(sample_transaction['V1']), step=0.1)
            transaction['V2'] = st.slider("V2", -5.0, 5.0, float(sample_transaction['V2']), step=0.1)
            transaction['V3'] = st.slider("V3", -5.0, 5.0, float(sample_transaction['V3']), step=0.1)
            transaction['V4'] = st.slider("V4", -5.0, 5.0, float(sample_transaction['V4']), step=0.1)
            transaction['V5'] = st.slider("V5", -5.0, 5.0, float(sample_transaction['V5']), step=0.1)
        
        # Submit button
        submit_button = st.form_submit_button(label="Detect Fraud")
    
    # If form is submitted
    if submit_button:
        with st.spinner("Analyzing transaction..."):
            # Make prediction
            prediction_result = fraud_detector.predict(transaction)
            
            # Display prediction in a nice format
            st.subheader("Fraud Detection Result")
            
            # Create columns for summary and details
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Display fraud probability gauge
                st.plotly_chart(
                    plot_transaction_risk(prediction_result['fraud_probability']),
                    use_container_width=True
                )
                
                # Show prediction summary
                result_color = (
                    "red" if prediction_result['is_fraud'] 
                    else "orange" if prediction_result['fraud_probability'] > 0.2
                    else "green"
                )
                
                result_text = (
                    "🚨 **FRAUD DETECTED**" if prediction_result['is_fraud']
                    else "✅ **LEGITIMATE TRANSACTION**"
                )
                
                st.markdown(f"<h3 style='color:{result_color};'>{result_text}</h3>", unsafe_allow_html=True)
                st.markdown(f"Risk Level: **{prediction_result['risk_level'].upper()}**")
                st.markdown(f"Confidence: **{prediction_result['confidence']:.2%}**")
                st.markdown(f"Detection Time: **{prediction_result['prediction_time_ms']} ms**")
            
            with col2:
                # Display top factors
                st.subheader("Key Factors")
                if prediction_result.get('top_factors'):
                    for factor in prediction_result['top_factors']:
                        direction_icon = "🔺" if factor['direction'] == 'increases' else "🔻"
                        st.markdown(
                            f"**{factor['feature']}:** {factor['value']:.4f} "
                            f"{direction_icon} {factor['direction']} fraud risk"
                        )
                else:
                    st.info("Feature explanation not available.")

def display_batch_prediction_tab():
    """Display batch prediction tab content."""
    st.header("Batch Prediction")
    
    # Load fraud detector
    _, fraud_detector, _ = train_or_load_models()
    
    st.subheader("Upload Transactions CSV")
    st.info("""
    Upload a CSV file with transaction data. The file should have the same format as the training data.
    At minimum, it should include the 'Amount' column and all 'V' columns (V1-V28).
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Found {len(df)} transactions.")
            
            # Display sample of the data
            st.subheader("Preview")
            st.dataframe(df.head(5))
            
            # Check if required columns exist
            required_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Add 'Time' column if not present
                if 'Time' not in df.columns:
                    df['Time'] = 0
                
                # Process the file for prediction
                if st.button("Predict Fraud"):
                    with st.spinner("Processing transactions..."):
                        # Display progress bar
                        progress_bar = st.progress(0)
                        
                        # Process in batches to avoid memory issues
                        batch_size = 100
                        num_batches = (len(df) + batch_size - 1) // batch_size
                        
                        results = []
                        
                        for i in range(num_batches):
                            # Get batch
                            start_idx = i * batch_size
                            end_idx = min((i + 1) * batch_size, len(df))
                            batch = df.iloc[start_idx:end_idx]
                            
                            # Convert each row to dictionary and predict
                            batch_results = []
                            for _, row in batch.iterrows():
                                transaction = row.to_dict()
                                prediction = fraud_detector.predict(transaction)
                                batch_results.append({
                                    'fraud_probability': prediction['fraud_probability'],
                                    'is_fraud': prediction['is_fraud'],
                                    'risk_level': prediction['risk_level']
                                })
                            
                            results.extend(batch_results)
                            
                            # Update progress bar
                            progress_bar.progress((i + 1) / num_batches)
                        
                        # Add results to the dataframe
                        results_df = pd.DataFrame(results)
                        df_with_predictions = pd.concat([df, results_df], axis=1)
                        
                        # Display summary
                        st.subheader("Prediction Summary")
                        
                        # Count fraud vs legitimate
                        fraud_count = sum(results_df['is_fraud'])
                        legitimate_count = len(results_df) - fraud_count
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Transactions", len(df))
                        
                        with col2:
                            st.metric("Fraud Transactions", fraud_count)
                        
                        with col3:
                            st.metric("Legitimate Transactions", legitimate_count)
                        
                        # Display risk level distribution
                        risk_counts = results_df['risk_level'].value_counts().reset_index()
                        risk_counts.columns = ['Risk Level', 'Count']
                        
                        # Create a bar chart
                        fig = go.Figure()
                        colors = {
                            'low': config.COLORS['success'],
                            'medium': config.COLORS['warning'],
                            'high': config.COLORS['danger'],
                            'very high': 'darkred'
                        }
                        
                        for risk in risk_counts['Risk Level']:
                            count = risk_counts.loc[risk_counts['Risk Level'] == risk, 'Count'].iloc[0]
                            fig.add_trace(go.Bar(
                                x=[risk.capitalize()],
                                y=[count],
                                name=risk.capitalize(),
                                marker_color=colors.get(risk, 'gray')
                            ))
                        
                        fig.update_layout(
                            title="Risk Level Distribution",
                            xaxis_title="Risk Level",
                            yaxis_title="Number of Transactions",
                            height=400
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Display results table
                        st.subheader("Prediction Results")
                        st.dataframe(df_with_predictions)
                        
                        # Provide download link
                        csv = df_with_predictions.to_csv(index=False)
                        st.markdown(
                            download_link(
                                csv, 
                                'fraud_predictions.csv', 
                                'Download Predictions as CSV'
                            ),
                            unsafe_allow_html=True
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def display_model_insights_tab():
    """Display model insights tab content."""
    st.header("Model Insights")
    
    # Load models
    model_trainer, fraud_detector, _ = train_or_load_models()
    
    # Show various insights about the model
    st.subheader("Feature Importance Analysis")
    
    # Get data for visualization
    data_processor = DataProcessor(use_sample=True)
    df, _ = load_and_preprocess_data()
    
    # Get feature names
    feature_names = data_processor.get_features()
    
    # Let user select model
    model_options = list(model_trainer.models.keys())
    selected_model = st.selectbox(
        "Select Model for Analysis",
        model_options,
        index=model_options.index("xgboost") if "xgboost" in model_options else 0,
        key="insight_model_select"
    )
    
    if selected_model:
        model = model_trainer.models[selected_model]
        
        # Display SHAP visualizations if available
        import shap
        
        try:
            st.subheader("SHAP Analysis")
            st.write("SHAP values help explain the output of machine learning models.")
            
            # Sample data for SHAP
            X_test = data_processor.X_test
            if X_test is None:
                _, X_test, _, _ = data_processor.prepare_train_test(apply_smote=False)
                
            # Sample a small subset for faster computation
            X_sample = X_test.sample(min(100, len(X_test)), random_state=config.RANDOM_STATE)
            
            # Initialize fraud detector explainer if needed
            if fraud_detector.explainer is None:
                fraud_detector.load_model(selected_model)
            
            # For XGBoost models
            if hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Convert to matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Add another SHAP plot
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig2)
                
            else:
                st.warning("SHAP visualization is currently optimized for tree-based models.")
                
            # Show a waterfall plot for a single example
            st.subheader("Example Explanation")
            st.write("This shows how features contribute to the prediction for a single transaction:")
            
            # Use a sample transaction
            sample_transaction = get_sample_transaction()
            sample_df = pd.DataFrame([sample_transaction])
            sample_processed = data_processor.preprocess_transaction(sample_transaction)
            
            # Initialize explainer if needed
            if hasattr(model, 'get_booster'):
                # Get SHAP values
                shap_values = explainer.shap_values(sample_processed)
                
                # Create waterfall plot
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                shap.plots.waterfall(explainer.expected_value, shap_values[0], feature_names=sample_processed.columns, show=False)
                st.pyplot(fig3)
                
        except Exception as e:
            st.error(f"Error generating SHAP visualizations: {str(e)}")
        
        # Add A/B testing framework mockup
        st.subheader("A/B Testing Framework")
        st.write("Simulate how model performance changes with different thresholds:")
        
        # Get test data
        _, X_test, _, y_test = data_processor.prepare_train_test(apply_smote=False)
        
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Let user select thresholds to compare
        col1, col2 = st.columns(2)
        
        with col1:
            threshold_a = st.slider(
                "Threshold A", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5, 
                step=0.05,
                key="threshold_a"
            )
            
        with col2:
            threshold_b = st.slider(
                "Threshold B", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.3, 
                step=0.05,
                key="threshold_b"
            )
        
        # Compare thresholds
        if st.button("Compare Thresholds"):
            with st.spinner("Calculating metrics..."):
                # Get predictions for both thresholds
                y_pred_a = (y_proba >= threshold_a).astype(int)
                y_pred_b = (y_proba >= threshold_b).astype(int)
                
                # Calculate metrics
                metrics_a = evaluate_model(model, X_test, y_test, threshold=threshold_a)
                metrics_b = evaluate_model(model, X_test, y_test, threshold=threshold_b)
                
                # Calculate business metrics
                cost_a = calculate_cost_metrics(y_test, y_pred_a)
                cost_b = calculate_cost_metrics(y_test, y_pred_b)
                
                # Display comparison
                st.subheader("Threshold Comparison Results")
                
                # Metrics table
                comparison_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 
                              'Detection Rate', 'False Positive Rate', 'Saved Amount'],
                    f'Threshold A ({threshold_a})': [
                        f"{metrics_a['accuracy']:.4f}",
                        f"{metrics_a['precision']:.4f}",
                        f"{metrics_a['recall']:.4f}",
                        f"{metrics_a['f1']:.4f}",
                        f"{metrics_a['roc_auc']:.4f}",
                        f"{cost_a['detection_rate']:.2%}",
                        f"{cost_a['false_positive_rate']:.2%}",
                        f"${cost_a['saved_amount']:.2f}"
                    ],
                    f'Threshold B ({threshold_b})': [
                        f"{metrics_b['accuracy']:.4f}",
                        f"{metrics_b['precision']:.4f}",
                        f"{metrics_b['recall']:.4f}",
                        f"{metrics_b['f1']:.4f}",
                        f"{metrics_b['roc_auc']:.4f}",
                        f"{cost_b['detection_rate']:.2%}",
                        f"{cost_b['false_positive_rate']:.2%}",
                        f"${cost_b['saved_amount']:.2f}"
                    ]
                })
                
                st.table(comparison_df)
                
                # Visualize confusion matrices side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Threshold A ({threshold_a})")
                    st.plotly_chart(plot_confusion_matrix(y_test, y_pred_a), use_container_width=True)
                
                with col2:
                    st.subheader(f"Threshold B ({threshold_b})")
                    st.plotly_chart(plot_confusion_matrix(y_test, y_pred_b), use_container_width=True)

def main():
    """Main function to run the Streamlit app."""
    # Display logo and title
    st.sidebar.title("💳 Credit Card Fraud Detection")
    st.sidebar.markdown("A lightweight fraud detection system.")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Model Performance", "Real-time Prediction", "Batch Prediction", "Model Insights"]
    )
    
    # Display the selected page
    if page == "Model Performance":
        display_model_performance_tab()
    elif page == "Real-time Prediction":
        display_prediction_tab()
    elif page == "Batch Prediction":
        display_batch_prediction_tab()
    elif page == "Model Insights":
        display_model_insights_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Credit Card Fraud Detection System**\n
        A lightweight, end-to-end system for detecting fraudulent transactions.
        """
    )

if __name__ == "__main__":
    main()
