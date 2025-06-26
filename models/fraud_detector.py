"""
Fraud detection engine for credit card transactions.
Handles real-time prediction and confidence scoring.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
import joblib
import os
import time
import shap
from functools import lru_cache

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
import config
from models.model_trainer import ModelTrainer
from data.data_processor import DataProcessor
from utils.metrics import get_risk_levels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudDetector:
    """Engine for detecting fraudulent credit card transactions."""
    
    def __init__(self, use_sample_data: bool = False):
        """
        Initialize the fraud detector.
        
        Args:
            use_sample_data (bool, optional): Whether to use sample data for demo. Defaults to False.
        """
        self.model_trainer = ModelTrainer()
        self.data_processor = DataProcessor(use_sample=use_sample_data)
        self.model = None
        self.threshold = 0.5
        self.feature_names = None
        self.explainer = None
    
    def load_model(self, model_name: str = None) -> None:
        """
        Load the model and its threshold.
        
        Args:
            model_name (str, optional): Name of the model to load. Defaults to best model.
        """
        if model_name is None:
            # Load best model
            self.model, self.threshold = self.model_trainer.get_best_model()
        else:
            # Load specific model
            self.model_trainer.load_models()
            if model_name in self.model_trainer.models:
                self.model = self.model_trainer.models[model_name]
                
                # Load its threshold if available
                best_model_info_path = os.path.join(config.MODELS_DIR, "best_model_info.pkl")
                if os.path.exists(best_model_info_path):
                    best_model_info = joblib.load(best_model_info_path)
                    if best_model_info["name"] == model_name:
                        self.threshold = best_model_info["threshold"]
            else:
                raise ValueError(f"Model {model_name} not found")
        
        # Load feature names
        self.feature_names = self.model_trainer.get_feature_names()
        
        # Initialize explainer for the model
        self._init_explainer()
    
    def _init_explainer(self) -> None:
        """Initialize the SHAP explainer for the model."""
        logger.info("Initializing SHAP explainer")
        
        # Load a small subset of data for the explainer
        # This is to avoid loading the full dataset which could be large
        try:
            # Try to use existing test data if available
            if hasattr(self.data_processor, 'X_test') and self.data_processor.X_test is not None:
                background_data = self.data_processor.X_test.sample(
                    min(100, len(self.data_processor.X_test)),
                    random_state=config.RANDOM_STATE
                )
            else:
                # Otherwise load and process data
                df = self.data_processor.load_data()
                df = self.data_processor.preprocess_data(df)
                X = df.drop([config.TARGET_COLUMN], axis=1)
                X = self.data_processor.scale_features(X)
                background_data = X.sample(
                    min(100, len(X)),
                    random_state=config.RANDOM_STATE
                )
            
            # Initialize the explainer based on model type
            if hasattr(self.model, 'get_booster'):
                # XGBoost model
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Other models
                self.explainer = shap.Explainer(self.model, background_data)
                
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None
    
    def predict(self, transaction: Dict) -> Dict[str, Any]:
        """
        Predict whether a transaction is fraudulent.
        
        Args:
            transaction (Dict): Dictionary containing transaction features
            
        Returns:
            Dict[str, Any]: Prediction result with risk score and confidence
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # Preprocess transaction
        transaction_df = self.data_processor.preprocess_transaction(transaction)
        
        # Make prediction
        fraud_proba = self.model.predict_proba(transaction_df)[0, 1]
        is_fraud = (fraud_proba >= self.threshold)
        risk_level = get_risk_levels([fraud_proba])[0]
        
        # Calculate confidence level (scaled probability)
        if is_fraud:
            confidence = (fraud_proba - self.threshold) / (1 - self.threshold)
        else:
            confidence = (self.threshold - fraud_proba) / self.threshold
        
        # Limit confidence to 0.5-1.0 range
        confidence = 0.5 + (confidence / 2)
        
        # Get top factors only if explainer is available
        top_factors = self.get_prediction_factors(transaction_df) if self.explainer else None
        
        prediction_time = time.time() - start_time
        
        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(fraud_proba),
            "risk_level": risk_level,
            "confidence": float(confidence),
            "threshold": float(self.threshold),
            "top_factors": top_factors,
            "prediction_time_ms": round(prediction_time * 1000, 2)
        }
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict[str, Any]]:
        """
        Predict whether multiple transactions are fraudulent.
        
        Args:
            transactions (List[Dict]): List of transaction dictionaries
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        results = []
        for transaction in transactions:
            results.append(self.predict(transaction))
        
        return results
    
    def get_prediction_factors(self, transaction_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top factors influencing the prediction.
        
        Args:
            transaction_df (pd.DataFrame): Preprocessed transaction data
            top_n (int, optional): Number of top factors to return. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: List of top factors with feature name, value, and importance
        """
        if self.explainer is None:
            return None
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(transaction_df)
            
            # For models that return a list of shap values (one per class)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Get values for positive class
            
            # Create a DataFrame with feature names and their SHAP values
            factors_df = pd.DataFrame({
                'feature': transaction_df.columns,
                'value': transaction_df.values[0],
                'importance': np.abs(shap_values[0]),
                'impact': shap_values[0]
            })
            
            # Sort by absolute importance and get top N
            factors_df = factors_df.sort_values('importance', ascending=False).head(top_n)
            
            # Determine if impact is positive (increases fraud likelihood) or negative
            factors_df['direction'] = factors_df['impact'].apply(lambda x: 'increases' if x > 0 else 'decreases')
            
            # Convert to list of dictionaries
            top_factors = factors_df.to_dict('records')
            
            return top_factors
        
        except Exception as e:
            logger.error(f"Error calculating prediction factors: {e}")
            return None
    
    @lru_cache(maxsize=128)
    def get_shap_values(self, transaction_id: str, transaction_data: tuple) -> np.ndarray:
        """
        Get SHAP values for a transaction with caching.
        
        Args:
            transaction_id (str): Unique ID for the transaction (for caching)
            transaction_data (tuple): Transaction data as a tuple (for caching)
            
        Returns:
            np.ndarray: SHAP values for the transaction
        """
        if self.explainer is None:
            return None
        
        # Convert tuple back to DataFrame
        transaction_df = pd.DataFrame([transaction_data], columns=self.feature_names)
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(transaction_df)
            
            # For models that return a list of shap values (one per class)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Get values for positive class
            
            return shap_values[0]
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None 