"""
Data processing pipeline for credit card fraud detection.
Handles loading, cleaning, feature engineering, and preprocessing.
"""
import os
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import requests
from imblearn.over_sampling import SMOTE
import logging
import datetime
import joblib
from pathlib import Path
import sys

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process credit card transaction data for fraud detection."""
    
    def __init__(self, use_sample: bool = False):
        """
        Initialize the data processor.
        
        Args:
            use_sample (bool): Whether to use a sample dataset for quick demos.
        """
        self.use_sample = use_sample
        self.data_path = config.SAMPLE_DATA_PATH if use_sample else config.LOCAL_DATA_PATH
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def download_dataset(self) -> None:
        """Download the credit card fraud dataset if not already present."""
        if os.path.exists(self.data_path):
            logger.info(f"Dataset already exists at {self.data_path}")
            return
        
        logger.info(f"Downloading dataset from {config.DATASET_URL}")
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            # Download with a streaming approach to handle large files
            with requests.get(config.DATASET_URL, stream=True) as r:
                r.raise_for_status()
                with open(self.data_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Dataset downloaded successfully to {self.data_path}")
            
            # Create sample data for demo purposes
            if not os.path.exists(config.SAMPLE_DATA_PATH):
                self._create_sample_dataset()
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def _create_sample_dataset(self) -> None:
        """Create a small sample dataset for demos."""
        try:
            df = pd.read_csv(config.LOCAL_DATA_PATH)
            
            # Get all fraud cases
            fraud_df = df[df[config.TARGET_COLUMN] == 1]
            
            # Get a random sample of non-fraud cases (10x the fraud cases)
            non_fraud_df = df[df[config.TARGET_COLUMN] == 0].sample(
                n=min(len(fraud_df) * 10, len(df[df[config.TARGET_COLUMN] == 0])),
                random_state=config.RANDOM_STATE
            )
            
            # Combine and shuffle
            sample_df = pd.concat([fraud_df, non_fraud_df]).sample(
                frac=1, random_state=config.RANDOM_STATE
            ).reset_index(drop=True)
            
            # Save to sample file
            sample_df.to_csv(config.SAMPLE_DATA_PATH, index=False)
            logger.info(f"Created sample dataset with {len(sample_df)} records")
            
        except Exception as e:
            logger.error(f"Error creating sample dataset: {e}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the credit card transaction dataset.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if not os.path.exists(self.data_path):
            self.download_dataset()
        
        logger.info(f"Loading data from {self.data_path}")
        try:
            # Use optimized dtypes for memory efficiency
            df = pd.read_csv(self.data_path, dtype={
                'Time': 'float32',
                'Amount': 'float32',
                'Class': 'uint8'
            })
            
            # Use float32 for V columns to save memory
            for col in df.columns:
                if col.startswith('V'):
                    df[col] = df[col].astype('float32')
            
            logger.info(f"Loaded dataset with shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values, outliers, and feature engineering.
        
        Args:
            df (pd.DataFrame): The raw data.
            
        Returns:
            pd.DataFrame: The preprocessed data.
        """
        logger.info("Preprocessing data")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Found {missing_values.sum()} missing values")
            # Fill missing values with median for numerical features
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Handle outliers in Amount using capping
        df = self._handle_outliers(df, 'Amount')
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features to improve model performance.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with engineered features.
        """
        # Convert time to more meaningful features
        # Time is in seconds from the start of the dataset
        if 'Time' in df.columns:
            # Normalize time to hours
            df['hour'] = (df['Time'] / 3600) % 24
            
            # Assuming the dataset spans multiple days, create day features
            df['day'] = (df['Time'] / (3600 * 24)).astype(int)
            
            # Create day of week feature (0-6, where 0 is Monday)
            df['day_of_week'] = df['day'] % 7
            
            # Create weekend indicator
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # We can drop the original Time column if we don't need it
            # df = df.drop('Time', axis=1)
        
        # Ratio of transaction amount to mean amount (risk indicator)
        df['amount_to_mean_ratio'] = df['Amount'] / df['Amount'].mean()
        
        # Log transform of amount to handle skewness
        df['log_amount'] = np.log1p(df['Amount'])
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Handle outliers in a specified column using capping.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            column (str): The column to handle outliers for.
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled.
        """
        Q1 = df[column].quantile(0.01)
        Q3 = df[column].quantile(0.99)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        df[column] = df[column].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            fit (bool): Whether to fit the scaler or just transform.
            
        Returns:
            pd.DataFrame: Dataframe with scaled features.
        """
        # Identify columns to scale
        cols_to_scale = config.FEATURES_TO_SCALE.copy()
        
        # Add V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        cols_to_scale.extend(v_cols)
        
        # Create a copy of the dataframe
        scaled_df = df.copy()
        
        if fit:
            # Fit and transform
            scaled_df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            
            # Save the scaler for future use
            joblib.dump(self.scaler, os.path.join(config.MODELS_DIR, 'scaler.pkl'))
        else:
            # Just transform
            scaled_df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return scaled_df
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled features and target.
        """
        logger.info("Applying SMOTE to handle class imbalance")
        
        # Check class distribution before SMOTE
        class_counts = y.value_counts()
        logger.info(f"Class distribution before SMOTE: {dict(class_counts)}")
        
        # Check if we have enough samples for SMOTE
        min_samples_needed = 6  # SMOTE typically needs at least 6 samples of minority class
        if class_counts.min() < min_samples_needed:
            logger.warning(f"Not enough samples for SMOTE (minimum {min_samples_needed} needed). Using simple duplication instead.")
            
            # Identify minority class
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            
            # Get indices for each class
            minority_indices = y[y == minority_class].index
            
            # Calculate how many duplicates we need
            target_ratio = config.SMOTE_PARAMS.get("sampling_strategy", 0.1)
            target_minority_count = int(class_counts[majority_class] * target_ratio)
            duplicates_needed = max(0, target_minority_count - class_counts[minority_class])
            
            if duplicates_needed > 0:
                # Duplicate minority samples with small random variations
                minority_X = X.loc[minority_indices].copy()
                minority_y = y.loc[minority_indices].copy()
                
                # Create duplicates with small variations
                duplicates_X = pd.DataFrame()
                duplicates_y = pd.Series(dtype=y.dtype)
                
                for _ in range(duplicates_needed):
                    # Select a random sample to duplicate
                    idx = np.random.choice(minority_indices)
                    sample_X = X.loc[idx:idx].copy()
                    sample_y = y.loc[idx:idx].copy()
                    
                    # Add small random variations (1-2% of original values)
                    for col in sample_X.columns:
                        noise = np.random.normal(0, 0.01 * abs(sample_X[col].values[0]) + 0.001)
                        sample_X[col] = sample_X[col] + noise
                    
                    # Add to duplicates
                    duplicates_X = pd.concat([duplicates_X, sample_X])
                    duplicates_y = pd.concat([duplicates_y, sample_y])
                
                # Combine original and duplicated data
                X_resampled = pd.concat([X, duplicates_X])
                y_resampled = pd.concat([y, duplicates_y])
            else:
                X_resampled, y_resampled = X, y
        else:
            # Apply SMOTE
            smote = SMOTE(**config.SMOTE_PARAMS)
            X_resampled_array, y_resampled_array = smote.fit_resample(X, y)
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled_array, columns=X.columns)
            y_resampled = pd.Series(y_resampled_array, name=y.name)
        
        # Check class distribution after resampling
        logger.info(f"Class distribution after resampling: {dict(pd.Series(y_resampled).value_counts())}")
        
        return X_resampled, y_resampled
    
    def prepare_train_test(self, test_size: float = None, apply_smote: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train and test datasets.
        
        Args:
            test_size (float, optional): Size of test set. Defaults to config.TEST_SIZE.
            apply_smote (bool, optional): Whether to apply SMOTE to the training data. Defaults to True.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        
        # Load and preprocess data
        df = self.load_data()
        df = self.preprocess_data(df)
        
        # Split features and target
        X = df.drop([config.TARGET_COLUMN], axis=1)
        y = df[config.TARGET_COLUMN]
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Check if we have enough samples of each class for stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        # Split into train and test
        try:
            # Try stratified split first
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using regular split instead.")
            # Fall back to regular split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=None
            )
            
            # Check if we have both classes in train and test sets
            if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
                logger.warning("One of the splits doesn't have both classes. Adjusting split manually.")
                # Ensure both train and test sets have at least one sample of each class
                fraud_indices = y[y == 1].index
                non_fraud_indices = y[y == 0].index
                
                # Calculate how many of each to put in test set
                n_fraud_test = max(1, int(len(fraud_indices) * test_size))
                n_non_fraud_test = max(1, int(len(non_fraud_indices) * test_size))
                
                # Split indices
                fraud_test_indices = fraud_indices[:n_fraud_test]
                fraud_train_indices = fraud_indices[n_fraud_test:]
                non_fraud_test_indices = non_fraud_indices[:n_non_fraud_test]
                non_fraud_train_indices = non_fraud_indices[n_non_fraud_test:]
                
                # Combine indices
                test_indices = pd.concat([fraud_test_indices, non_fraud_test_indices])
                train_indices = pd.concat([fraud_train_indices, non_fraud_train_indices])
                
                # Create new splits
                X_train, X_test = X.loc[train_indices], X.loc[test_indices]
                y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        
        # Scale features
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        # Apply SMOTE to training data if requested
        if apply_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # Store the datasets for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        logger.info(f"Train set fraud cases: {sum(y_train)}, Test set fraud cases: {sum(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_features(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names
    
    def preprocess_transaction(self, transaction: Dict) -> pd.DataFrame:
        """
        Preprocess a single transaction for prediction.
        
        Args:
            transaction (Dict): A dictionary containing transaction features.
            
        Returns:
            pd.DataFrame: Preprocessed transaction ready for prediction.
        """
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction])
        
        # Make sure we have all features
        for feature in self.feature_names:
            if feature not in df.columns and feature not in ['hour', 'day', 'day_of_week', 'is_weekend', 'amount_to_mean_ratio', 'log_amount']:
                df[feature] = 0
        
        # Apply feature engineering
        df = self._engineer_features(df)
        
        # Scale features
        # Load scaler if not already loaded
        if not hasattr(self, 'scaler') or self.scaler is None:
            scaler_path = os.path.join(config.MODELS_DIR, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError("Scaler model not found. Please train the model first.")
        
        df = self.scale_features(df, fit=False)
        
        # Select only the features used by the model
        return df[self.feature_names]