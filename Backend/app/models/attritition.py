# path: mindcare-backend/app/models/attrition.py
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app import config
from app.storage import storage
from app.utils.timers import timed

logger = logging.getLogger(__name__)

# Try to import XGBoost, but don't require it
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost is available for modeling")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost not available, will use scikit-learn models only")

class AttritionModel:
    """Attrition prediction model"""
    
    def __init__(self, model_type: str = "logistic_regression"):
        """
        Initialize the attrition model.
        
        Args:
            model_type: Type of model to use ("logistic_regression" or "xgboost")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_path = os.path.join(config.ARTIFACTS_DIR, f"attrition_{model_type}.joblib")
        self.scaler_path = os.path.join(config.ARTIFACTS_DIR, f"scaler_{model_type}.joblib")
        self.metadata_path = os.path.join(config.ARTIFACTS_DIR, f"metadata_{model_type}.joblib")
        
        # Ensure artifacts directory exists
        Path(config.ARTIFACTS_DIR).mkdir(exist_ok=True)
    
    def train(self, df_features: pd.DataFrame, target_col: str = "left_company") -> Tuple[bool, str]:
        """
        Train the attrition model.
        
        Args:
            df_features: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if target column exists
            if target_col not in df_features.columns:
                return False, f"Target column '{target_col}' not found in features"
            
            # Prepare features and target
            X = df_features.drop(columns=[target_col, 'employee_id'])
            y = df_features[target_col]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize model
            if self.model_type == "logistic_regression":
                self.model = LogisticRegression(
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                )
            elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            else:
                return False, f"Model type '{self.model_type}' is not supported"
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_proba)
            
            logger.info(f"Model training completed. AUC: {auc:.4f}")
            logger.info(f"Classification report: {report}")
            
            # Save model and metadata
            self._save_model()
            self._save_metadata({
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "auc": auc,
                "classification_report": report
            })
            
            return True, f"Model trained successfully with AUC: {auc:.4f}"
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False, f"Error training model: {str(e)}"
    
    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            df_features: DataFrame with features
            
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            if not self._load_model():
                logger.error("Model not trained and could not be loaded")
                return np.zeros(len(df_features))
        
        try:
            # Prepare features
            X = df_features[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_proba = self.model.predict_proba(X_scaled)[:, 1]
            
            return y_proba
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(df_features))
    
    def _save_model(self) -> bool:
        """Save the trained model and scaler"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def _load_model(self) -> bool:
        """Load the trained model and scaler"""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                return False
            
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Load metadata
            metadata = self._load_metadata()
            if metadata:
                self.feature_names = metadata.get("feature_names", [])
            
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_metadata(self, metadata: dict) -> bool:
        """Save model metadata"""
        try:
            joblib.dump(metadata, self.metadata_path)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def _load_metadata(self) -> dict:
        """Load model metadata"""
        try:
            if os.path.exists(self.metadata_path):
                return joblib.load(self.metadata_path)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return {}

@timed
def train_model(model_type: str = "logistic_regression") -> Tuple[bool, str]:
    """
    Train an attrition model.
    
    Args:
        model_type: Type of model to train
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Load features from database
        if not storage.table_exists('derived_employee_features'):
            return False, "Feature table does not exist. Run feature building first."
        
        df_features = storage.read_table('derived_employee_features')
        
        # Check if target column exists
        if 'left_company' not in df_features.columns:
            return False, "Target column 'left_company' not found in features"
        
        # Initialize and train model
        model = AttritionModel(model_type)
        success, message = model.train(df_features)
        
        return success, message
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False, f"Error training model: {str(e)}"

@timed
def predict_attrition(model_type: str = "logistic_regression", employee_ids: List[str] = None) -> pd.DataFrame:
    """
    Predict attrition for employees.
    
    Args:
        model_type: Type of model to use
        employee_ids: List of employee IDs to predict for. If None, predict for all.
        
    Returns:
        DataFrame with predictions
    """
    try:
        # Load features from database
        if not storage.table_exists('derived_employee_features'):
            logger.error("Feature table does not exist")
            return pd.DataFrame()
        
        df_features = storage.read_table('derived_employee_features')
        
        # Filter by employee IDs if provided
        if employee_ids:
            df_features = df_features[df_features['employee_id'].isin(employee_ids)]
        
        if len(df_features) == 0:
            logger.warning("No employees found for prediction")
            return pd.DataFrame()
        
        # Initialize model
        model = AttritionModel(model_type)
        
        # Make predictions
        predictions = model.predict(df_features)
        
        # Create result DataFrame
        result_df = df_features[['employee_id']].copy()
        result_df['attrition_risk'] = predictions
        
        logger.info(f"Generated attrition predictions for {len(result_df)} employees")
        return result_df
    except Exception as e:
        logger.error(f"Error predicting attrition: {str(e)}")
        return pd.DataFrame()