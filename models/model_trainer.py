"""
Model training module for credit card fraud detection.
Handles training, validation, and model persistence.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb
import joblib
import os
import logging
import sys
from pathlib import Path
import time

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils.metrics import evaluate_model, find_optimal_threshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and validates fraud detection models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.models_metrics = {}
        self.best_model_name = None
        self.best_model = None
        self.best_threshold = 0.5
        self.feature_names = None
        
    def get_models_to_train(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dict[str, Any]: Dictionary of model name to model instance
        """
        models = {
            "logistic_regression": LogisticRegression(**config.MODEL_PARAMS["logistic_regression"]),
            "random_forest": RandomForestClassifier(**config.MODEL_PARAMS["random_forest"]),
            "xgboost": xgb.XGBClassifier(**config.MODEL_PARAMS["xgboost"])
        }
        
        # Create voting classifier
        voting_models = [
            ('lr', models['logistic_regression']),
            ('rf', models['random_forest']),
            ('xgb', models['xgboost'])
        ]
        models["voting"] = VotingClassifier(
            estimators=voting_models,
            **config.MODEL_PARAMS["voting"]
        )
        
        return models
    
    def cross_validate_model(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv: int = None
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Dictionary of cross-validation metrics
        """
        if cv is None:
            cv = config.CV_FOLDS
        
        logger.info(f"Cross-validating {model.__class__.__name__}")
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        start_time = time.time()
        cv_results = cross_validate(
            model, X, y, cv=skf, scoring=scoring, return_train_score=False
        )
        end_time = time.time()
        
        # Extract and format results
        metrics = {}
        for metric in scoring.keys():
            metrics[metric] = np.mean(cv_results[f'test_{metric}'])
        
        # Add time taken
        metrics['time'] = end_time - start_time
        
        return metrics
    
    def train_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary with model names and their metrics
        """
        logger.info("Training models")
        
        # Save feature names
        self.feature_names = X_train.columns.tolist()
        
        # Get models to train
        models_to_train = self.get_models_to_train()
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            logger.info(f"Training {name}")
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Save model
            self.models[name] = model
            
            # Cross-validate model
            cv_metrics = self.cross_validate_model(model, X_train, y_train)
            
            # Evaluate on test set if provided
            if X_test is not None and y_test is not None:
                # Find optimal threshold based on F1 score
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    threshold = find_optimal_threshold(y_test, y_proba)
                else:
                    threshold = 0.5
                
                # Evaluate with optimal threshold
                test_metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
                test_metrics['optimal_threshold'] = threshold
            else:
                test_metrics = {}
            
            # Combine metrics
            self.models_metrics[name] = {
                **cv_metrics,
                **test_metrics,
                'train_time': train_time
            }
            
            logger.info(f"{name} metrics: {self.models_metrics[name]}")
        
        # Determine best model based on F1 score
        if self.models_metrics:
            best_model_name = max(self.models_metrics, key=lambda k: self.models_metrics[k].get('f1', 0))
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            # Get optimal threshold for best model
            if 'optimal_threshold' in self.models_metrics[best_model_name]:
                self.best_threshold = self.models_metrics[best_model_name]['optimal_threshold']
            
            logger.info(f"Best model: {best_model_name} with F1 score: {self.models_metrics[best_model_name].get('f1', 'N/A')}")
        
        return self.models_metrics
    
    def save_models(self, compress: bool = True) -> None:
        """
        Save trained models to disk.
        
        Args:
            compress (bool, optional): Whether to compress the saved models. Defaults to True.
        """
        logger.info("Saving models")
        
        # Create models directory if it doesn't exist
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = os.path.join(config.MODELS_DIR, f"{name}_model.pkl")
            
            # Use compression for smaller file size
            compress_level = 9 if compress else 0
            
            joblib.dump(
                model, 
                model_path, 
                compress=compress_level
            )
            
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save feature names
        if self.feature_names is not None:
            feature_names_path = os.path.join(config.MODELS_DIR, "feature_names.pkl")
            joblib.dump(self.feature_names, feature_names_path)
            
        # Save best model info
        if self.best_model_name is not None:
            best_model_info = {
                "name": self.best_model_name,
                "threshold": self.best_threshold
            }
            best_model_info_path = os.path.join(config.MODELS_DIR, "best_model_info.pkl")
            joblib.dump(best_model_info, best_model_info_path)
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load trained models from disk.
        
        Returns:
            Dict[str, Any]: Dictionary of loaded models
        """
        logger.info("Loading models")
        
        # Create models directory if it doesn't exist
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Load models
        models = {}
        model_files = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            model_path = os.path.join(config.MODELS_DIR, model_file)
            
            try:
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
        
        self.models = models
        
        # Load feature names
        feature_names_path = os.path.join(config.MODELS_DIR, "feature_names.pkl")
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
        
        # Load best model info
        best_model_info_path = os.path.join(config.MODELS_DIR, "best_model_info.pkl")
        if os.path.exists(best_model_info_path):
            best_model_info = joblib.load(best_model_info_path)
            self.best_model_name = best_model_info["name"]
            self.best_threshold = best_model_info["threshold"]
            
            if self.best_model_name in self.models:
                self.best_model = self.models[self.best_model_name]
        
        return self.models
    
    def get_best_model(self) -> Tuple[Any, float]:
        """
        Get the best performing model and its threshold.
        
        Returns:
            Tuple[Any, float]: Best model and its optimal threshold
        """
        if self.best_model is None:
            if not self.models:
                self.load_models()
            
            if self.best_model_name is None and self.models:
                # Default to XGBoost if available
                if "xgboost" in self.models:
                    self.best_model_name = "xgboost"
                else:
                    # Otherwise, use the first available model
                    self.best_model_name = next(iter(self.models))
                
                self.best_model = self.models[self.best_model_name]
        
        return self.best_model, self.best_threshold
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        
        Returns:
            List[str]: List of feature names
        """
        if self.feature_names is None:
            feature_names_path = os.path.join(config.MODELS_DIR, "feature_names.pkl")
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
        
        return self.feature_names 