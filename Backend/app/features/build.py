# path: mindcare-backend/app/features/build.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from app import config
from app.nlp.sentiment import calculate_survey_sentiment, aggregate_sentiment_by_team
from app.storage import storage
from app.utils.timers import timed

logger = logging.getLogger(__name__)

@timed
def compute_tenure_weeks(employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tenure in weeks for each employee.
    
    Args:
        employees_df: DataFrame with employee data
        
    Returns:
        DataFrame with added tenure_weeks column
    """
    df_result = employees_df.copy()
    
    # If tenure_weeks is already present, use it
    if 'tenure_weeks' in df_result.columns:
        logger.info("Using existing tenure_weeks column")
        return df_result
    
    # If hire_date is present, compute tenure_weeks
    if 'hire_date' in df_result.columns:
        current_date = datetime.now()
        df_result['hire_date'] = pd.to_datetime(df_result['hire_date'], errors='coerce')
        df_result['tenure_weeks'] = ((current_date - df_result['hire_date']).dt.days / 7).round().astype(int)
    else:
        logger.warning("Neither tenure_weeks nor hire_date columns found")
        df_result['tenure_weeks'] = 0
    
    logger.info(f"Computed tenure_weeks for {len(df_result)} employees")
    return df_result

@timed
def compute_survey_sentiment_features(surveys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment features from survey data.
    
    Args:
        surveys_df: DataFrame with survey data
        
    Returns:
        DataFrame with added sentiment features
    """
    # Calculate sentiment for each survey response
    df_result = calculate_survey_sentiment(surveys_df)
    
    # Aggregate by employee and month
    employee_monthly_sentiment = df_result.groupby(['employee_id', 'month'])['survey_sentiment'].mean().reset_index()
    employee_monthly_sentiment.columns = ['employee_id', 'month', 'avg_monthly_sentiment']
    
    # Merge back to original dataframe
    df_result = df_result.merge(employee_monthly_sentiment, on=['employee_id', 'month'], how='left')
    
    logger.info(f"Computed sentiment features for {len(df_result)} survey responses")
    return df_result

@timed
def compute_workload_ratio(timesheets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute workload ratio from timesheet data.
    
    Args:
        timesheets_df: DataFrame with timesheet data
        
    Returns:
        DataFrame with added workload_ratio column
    """
    df_result = timesheets_df.copy()
    
    # Calculate workload ratio (hours vs. 40h baseline)
    baseline_hours = 40
    df_result['workload_ratio'] = df_result['hours'] / baseline_hours
    
    logger.info(f"Computed workload_ratio for {len(df_result)} timesheet entries")
    return df_result

@timed
def compute_manager_1on1_cadence(surveys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute manager 1-on-1 meeting cadence from survey data.
    
    Args:
        surveys_df: DataFrame with survey data
        
    Returns:
        DataFrame with manager 1-on-1 cadence features
    """
    # Extract mentions of manager meetings from free text
    df_result = surveys_df.copy()
    
    # Simple keyword-based extraction
    meeting_keywords = ['manager', '1:1', '1 on 1', 'meeting', 'check-in', 'check in']
    
    def has_meeting_mention(text):
        if not isinstance(text, str):
            return 0
        text = text.lower()
        return 1 if any(keyword in text for keyword in meeting_keywords) else 0
    
    df_result['manager_meeting_mention'] = df_result['free_text'].apply(has_meeting_mention)
    
    # Aggregate by employee
    employee_meeting_rate = df_result.groupby('employee_id')['manager_meeting_mention'].mean().reset_index()
    employee_meeting_rate.columns = ['employee_id', 'manager_1on1_cadence']
    
    # Merge back to original dataframe
    df_result = df_result.merge(employee_meeting_rate, on='employee_id', how='left')
    
    logger.info(f"Computed manager 1-on-1 cadence for {len(df_result)} employees")
    return df_result

@timed
def compute_pto_spike(timesheets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PTO spike from timesheet data.
    
    Args:
        timesheets_df: DataFrame with timesheet data
        
    Returns:
        DataFrame with added pto_spike column
    """
    df_result = timesheets_df.copy()
    
    # Simple heuristic: PTO spike if hours are significantly below baseline
    baseline_hours = 40
    pto_threshold = 0.5 * baseline_hours  # 50% of baseline
    
    df_result['pto_spike'] = (df_result['hours'] < pto_threshold).astype(int)
    
    # Aggregate by employee
    employee_pto_rate = df_result.groupby('employee_id')['pto_spike'].mean().reset_index()
    employee_pto_rate.columns = ['employee_id', 'pto_spike_rate']
    
    # Merge back to original dataframe
    df_result = df_result.merge(employee_pto_rate, on='employee_id', how='left')
    
    logger.info(f"Computed PTO spike for {len(df_result)} timesheet entries")
    return df_result

@timed
def compute_internal_moves(employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute internal moves for each employee.
    
    Args:
        employees_df: DataFrame with employee data
        
    Returns:
        DataFrame with added internal_moves column
    """
    df_result = employees_df.copy()
    
    # For demo purposes, we'll simulate internal moves based on tenure
    # In a real implementation, this would come from historical role changes
    np.random.seed(42)
    df_result['internal_moves'] = np.random.poisson(0.1 * (df_result['tenure_weeks'] / 52), size=len(df_result))
    
    logger.info(f"Computed internal_moves for {len(df_result)} employees")
    return df_result

@timed
def compute_skill_gap_score(employees_df: pd.DataFrame, skills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute skill gap score for each employee.
    
    Args:
        employees_df: DataFrame with employee data
        skills_df: DataFrame with employee skills
        
    Returns:
        DataFrame with added skill_gap_score column
    """
    # Define role-skill requirements matrix
    role_skill_requirements = {
        'Data Engineer': {'Python': 3, 'SQL': 3, 'dbt': 2, 'Airflow': 2, 'Snowflake': 2},
        'ML Engineer': {'Python': 4, 'TensorFlow': 3, 'PyTorch': 3, 'Scikit-learn': 3, 'SQL': 2},
        'Analyst': {'SQL': 3, 'Excel': 3, 'Tableau': 2, 'Python': 2, 'Statistics': 2},
        'Manager': {'Leadership': 4, 'Communication': 3, 'Project Management': 3, 'Excel': 2, 'SQL': 1}
    }
    
    # Merge employee data with skills
    employee_skills = employees_df.merge(skills_df, on='employee_id', how='left')
    
    # Pivot skills to wide format
    skills_wide = employee_skills.pivot_table(
        index='employee_id', 
        columns='skill', 
        values='level', 
        fill_value=0
    ).reset_index()
    
    # Merge back to employees
    df_result = employees_df.merge(skills_wide, on='employee_id', how='left')
    
    # Compute skill gap for each employee
    skill_gaps = []
    for _, row in df_result.iterrows():
        role = row['role']
        if role not in role_skill_requirements:
            # Default skill gap if role not in requirements
            skill_gaps.append(0.0)
            continue
            
        required_skills = role_skill_requirements[role]
        total_gap = 0
        total_required = 0
        
        for skill, required_level in required_skills.items():
            actual_level = row.get(skill, 0)
            gap = max(0, required_level - actual_level)
            total_gap += gap
            total_required += required_level
        
        # Normalize gap score
        skill_gap = total_gap / total_required if total_required > 0 else 0
        skill_gaps.append(skill_gap)
    
    df_result['skill_gap_score'] = skill_gaps
    
    logger.info(f"Computed skill_gap_score for {len(df_result)} employees")
    return df_result

@timed
def build_all_features() -> bool:
    """
    Build all features and save to database.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load data from database
        employees_df = storage.read_table('employees')
        surveys_df = storage.read_table('surveys')
        timesheets_df = storage.read_table('timesheets')
        skills_df = storage.read_table('skills')
        
        # Compute features
        employees_df = compute_tenure_weeks(employees_df)
        employees_df = compute_internal_moves(employees_df)
        employees_df = compute_skill_gap_score(employees_df, skills_df)
        
        surveys_df = compute_survey_sentiment_features(surveys_df)
        surveys_df = compute_manager_1on1_cadence(surveys_df)
        
        timesheets_df = compute_workload_ratio(timesheets_df)
        timesheets_df = compute_pto_spike(timesheets_df)
        
        # Create feature table
        # First, aggregate timesheet data by employee
        timesheet_features = timesheets_df.groupby('employee_id').agg({
            'workload_ratio': 'mean',
            'pto_spike_rate': 'mean'
        }).reset_index()
        
        # Aggregate survey data by employee
        survey_features = surveys_df.groupby('employee_id').agg({
            'survey_sentiment': 'mean',
            'manager_1on1_cadence': 'mean'
        }).reset_index()
        
        # Merge all features
        feature_df = employees_df[['employee_id', 'role', 'practice', 'tenure_weeks', 'internal_moves', 'skill_gap_score']]
        
        if not timesheet_features.empty:
            feature_df = feature_df.merge(timesheet_features, on='employee_id', how='left')
        
        if not survey_features.empty:
            feature_df = feature_df.merge(survey_features, on='employee_id', how='left')
        
        # Fill missing values
        feature_df = feature_df.fillna(0)
        
        # Add target variable (left_company)
        if 'left_company' in employees_df.columns:
            feature_df = feature_df.merge(
                employees_df[['employee_id', 'left_company']], 
                on='employee_id', 
                how='left'
            )
        
        # Save feature table to database
        if storage.DUCKDB_AVAILABLE:
            # DuckDB can directly create a table from a DataFrame
            storage.conn.execute("CREATE OR REPLACE TABLE derived_employee_features AS SELECT * FROM feature_df")
        else:
            # SQLite requires creating the table first
            feature_df.to_sql('derived_employee_features', storage.conn, if_exists='replace', index=False)
        
        # Create team risk aggregation
        team_risk_df = aggregate_team_risk(feature_df)
        
        # Save team risk table to database
        if storage.DUCKDB_AVAILABLE:
            storage.conn.execute("CREATE OR REPLACE TABLE derived_team_risk AS SELECT * FROM team_risk_df")
        else:
            team_risk_df.to_sql('derived_team_risk', storage.conn, if_exists='replace', index=False)
        
        logger.info("Feature building completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error building features: {str(e)}")
        return False

@timed
def aggregate_team_risk(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate risk by team/practice.
    
    Args:
        feature_df: DataFrame with employee features
        
    Returns:
        DataFrame with team risk aggregation
    """
    # For now, use a simple heuristic for risk score
    # In a real implementation, this would use the model predictions
    feature_df['risk_score'] = (
        0.2 * feature_df['skill_gap_score'] +
        0.2 * (1 - feature_df.get('survey_sentiment', 0.5)) +
        0.2 * np.clip(feature_df.get('workload_ratio', 1.0) - 1.0, 0, 1) +
        0.2 * feature_df.get('pto_spike_rate', 0) +
        0.1 * (1 - feature_df.get('manager_1on1_cadence', 0.5)) +
        0.1 * (feature_df['tenure_weeks'] < 52).astype(int)  # Less than 1 year
    )
    
    # Aggregate by practice
    team_risk = feature_df.groupby('practice').agg({
        'risk_score': 'mean',
        'skill_gap_score': 'mean',
        'survey_sentiment': 'mean',
        'workload_ratio': 'mean',
        'pto_spike_rate': 'mean',
        'manager_1on1_cadence': 'mean'
    }).reset_index()
    
    team_risk.columns = [
        'team', 'avg_risk', 'avg_skill_gap', 'avg_sentiment', 
        'avg_workload_ratio', 'avg_pto_spike', 'avg_manager_1on1'
    ]
    
    # Determine top drivers
    driver_columns = ['avg_skill_gap', 'avg_sentiment', 'avg_workload_ratio', 'avg_pto_spike', 'avg_manager_1on1']
    driver_names = ['skill_gap', 'sentiment', 'workload_ratio', 'pto_spike', 'manager_1on1']
    
    for _, row in team_risk.iterrows():
        # Get driver values (invert sentiment so higher values indicate higher risk)
        driver_values = [
            row['avg_skill_gap'],
            1 - row['avg_sentiment'],
            max(0, row['avg_workload_ratio'] - 1),
            row['avg_pto_spike'],
            1 - row['avg_manager_1on1']
        ]
        
        # Get top 3 drivers
        top_indices = np.argsort(driver_values)[-3:][::-1]
        top_drivers = [driver_names[i] for i in top_indices]
        
        # Add to dataframe
        for i, driver in enumerate(top_drivers, 1):
            team_risk.loc[row.name, f'top_driver_{i}'] = driver
    
    # Fill NaN values for top drivers
    for i in range(1, 4):
        if f'top_driver_{i}' not in team_risk.columns:
            team_risk[f'top_driver_{i}'] = None
        else:
            team_risk[f'top_driver_{i}'] = team_risk[f'top_driver_{i}'].fillna('unknown')
    
    logger.info(f"Aggregated risk for {len(team_risk)} teams")
    return team_risk