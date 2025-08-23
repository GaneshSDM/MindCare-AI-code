# path: mindcare-backend/app/models/explain.py
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from app import config
from app.models.attrition import AttritionModel
from app.utils.timers import timed

logger = logging.getLogger(__name__)

@timed
def get_feature_importance_lr(model: AttritionModel) -> pd.DataFrame:
    """
    Get feature importance from logistic regression model.
    
    Args:
        model: Trained logistic regression model
        
    Returns:
        DataFrame with feature importance
    """
    if model.model is None or model.model_type != "logistic_regression":
        logger.error("Model is not a trained logistic regression model")
        return pd.DataFrame()
    
    try:
        # Get coefficients
        coefficients = model.model.coef_[0]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': model.feature_names,
            'importance': np.abs(coefficients),
            'coefficient': coefficients
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info("Extracted feature importance from logistic regression model")
        return importance_df
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return pd.DataFrame()

@timed
def get_feature_importance_permutation(
    model: AttritionModel, 
    df_features: pd.DataFrame, 
    target_col: str = "left_company",
    n_repeats: int = 5
) -> pd.DataFrame:
    """
    Get feature importance using permutation importance.
    
    Args:
        model: Trained model
        df_features: DataFrame with features and target
        target_col: Name of the target column
        n_repeats: Number of times to permute each feature
        
    Returns:
        DataFrame with feature importance
    """
    if model.model is None:
        logger.error("Model is not trained")
        return pd.DataFrame()
    
    try:
        # Prepare features and target
        X = df_features[model.feature_names]
        y = df_features[target_col]
        
        # Scale features
        X_scaled = model.scaler.transform(X)
        
        # Calculate permutation importance
        result = permutation_importance(
            model.model, 
            X_scaled, 
            y, 
            n_repeats=n_repeats, 
            random_state=42
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': model.feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info("Calculated permutation feature importance")
        return importance_df
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {str(e)}")
        return pd.DataFrame()

@timed
def get_top_drivers(
    model: AttritionModel, 
    df_features: pd.DataFrame, 
    employee_id: str, 
    method: str = "auto"
) -> List[Dict[str, str]]:
    """
    Get top drivers of attrition risk for an employee.
    
    Args:
        model: Trained model
        df_features: DataFrame with features
        employee_id: Employee ID
        method: Method to use ("lr", "permutation", or "auto")
        
    Returns:
        List of top drivers with feature names and values
    """
    try:
        # Get employee features
        employee_data = df_features[df_features['employee_id'] == employee_id]
        
        if len(employee_data) == 0:
            logger.warning(f"Employee {employee_id} not found in features")
            return []
        
        employee_features = employee_data[model.feature_names].iloc[0]
        
        # Choose method
        if method == "auto":
            method = "lr" if model.model_type == "logistic_regression" else "permutation"
        
        # Get feature importance
        if method == "lr" and model.model_type == "logistic_regression":
            importance_df = get_feature_importance_lr(model)
        else:
            importance_df = get_feature_importance_permutation(model, df_features)
        
        if len(importance_df) == 0:
            logger.error("Could not calculate feature importance")
            return []
        
        # Get top features
        top_features = importance_df.head(5)['feature'].tolist()
        
        # Create driver information
        drivers = []
        for feature in top_features:
            feature_value = employee_features[feature]
            
            # Determine if high or low value is driving risk
            if model.model_type == "logistic_regression":
                coefficient = importance_df[importance_df['feature'] == feature]['coefficient'].values[0]
                direction = "high" if coefficient > 0 else "low"
            else:
                # For permutation importance, we need to determine direction based on feature values
                # This is a simplification - in a real implementation, we'd use SHAP or similar
                feature_values = df_features[feature]
                median_value = feature_values.median()
                direction = "high" if feature_value > median_value else "low"
            
            drivers.append({
                "feature": feature,
                "value": float(feature_value),
                "direction": direction,
                "description": f"{direction} {feature.replace('_', ' ')}"
            })
        
        logger.info(f"Extracted top drivers for employee {employee_id}")
        return drivers
    except Exception as e:
        logger.error(f"Error extracting top drivers: {str(e)}")
        return []

@timed
def explain_prediction(
    model: AttritionModel, 
    df_features: pd.DataFrame, 
    employee_id: str, 
    method: str = "auto"
) -> Dict:
    """
    Explain a prediction for an employee.
    
    Args:
        model: Trained model
        df_features: DataFrame with features
        employee_id: Employee ID
        method: Method to use ("lr", "permutation", or "auto")
        
    Returns:
        Dictionary with prediction explanation
    """
    try:
        # Get prediction
        employee_data = df_features[df_features['employee_id'] == employee_id]
        
        if len(employee_data) == 0:
            logger.warning(f"Employee {employee_id} not found in features")
            return {}
        
        # Get prediction probability
        prediction = model.predict(employee_data)[0]
        
        # Get top drivers
        drivers = get_top_drivers(model, df_features, employee_id, method)
        
        # Create explanation
        explanation = {
            "employee_id": employee_id,
            "attrition_risk": float(prediction),
            "risk_level": "high" if prediction > 0.7 else "medium" if prediction > 0.3 else "low",
            "top_drivers": drivers[:3]  # Top 3 drivers
        }
        
        logger.info(f"Generated explanation for employee {employee_id}")
        return explanation
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        return {}