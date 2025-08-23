# path: mindcare-backend/app/utils/pii.py
import hashlib
import logging
from typing import List, Union

import pandas as pd
from app import config

logger = logging.getLogger(__name__)

def hash_id(value: str, salt: str = None) -> str:
    """
    Hash an ID value using SHA-256 with a salt.
    
    Args:
        value: The ID value to hash
        salt: Optional salt value, defaults to config.PII_SALT
        
    Returns:
        Hashed ID string
    """
    if salt is None:
        salt = config.PII_SALT
    
    # Combine value and salt
    salted_value = f"{value}{salt}"
    
    # Hash using SHA-256
    hashed = hashlib.sha256(salted_value.encode()).hexdigest()
    
    # Return first 16 characters for brevity
    return f"H_{hashed[:16]}"

def mask_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Mask PII columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to mask
        
    Returns:
        DataFrame with masked columns
    """
    df_masked = df.copy()
    
    for col in columns:
        if col in df_masked.columns:
            if col.endswith('_id'):
                # Hash ID columns
                df_masked[col] = df_masked[col].apply(lambda x: hash_id(str(x)) if pd.notna(x) else x)
            elif 'email' in col.lower():
                # Mask email addresses
                df_masked[col] = df_masked[col].apply(
                    lambda x: f"masked_{hash_id(str(x))}@example.com" if pd.notna(x) else x
                )
            elif 'name' in col.lower():
                # Mask names
                df_masked[col] = df_masked[col].apply(
                    lambda x: f"masked_{hash_id(str(x))}" if pd.notna(x) else x
                )
            else:
                # Generic masking for other PII
                df_masked[col] = df_masked[col].apply(
                    lambda x: f"masked_{hash_id(str(x))}" if pd.notna(x) else x
                )
    
    logger.info(f"Masked PII columns: {columns}")
    return df_masked

def bucket_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Bucket date columns into time periods (year, quarter, month).
    
    Args:
        df: Input DataFrame
        date_columns: List of date column names to bucket
        
    Returns:
        DataFrame with bucketed date columns
    """
    df_bucketed = df.copy()
    
    for col in date_columns:
        if col in df_bucketed.columns:
            # Convert to datetime if not already
            df_bucketed[col] = pd.to_datetime(df_bucketed[col], errors='coerce')
            
            # Extract year, quarter, month
            df_bucketed[f"{col}_year"] = df_bucketed[col].dt.year
            df_bucketed[f"{col}_quarter"] = df_bucketed[col].dt.quarter
            df_bucketed[f"{col}_month"] = df_bucketed[col].dt.month
    
    logger.info(f"Bucketed date columns: {date_columns}")
    return df_bucketed