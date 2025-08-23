# path: mindcare-backend/app/nlp/themes.py
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from app.utils.timers import timed

logger = logging.getLogger(__name__)

@timed
def extract_themes(texts: List[str], n_topics: int = 5, max_features: int = 1000) -> Tuple[np.ndarray, List[str]]:
    """
    Extract themes from text using Non-Negative Matrix Factorization (NMF).
    
    Args:
        texts: List of text documents
        n_topics: Number of topics/themes to extract
        max_features: Maximum number of features for the vectorizer
        
    Returns:
        Tuple of (document-topic matrix, topic terms)
    """
    if not texts or len(texts) == 0:
        logger.warning("No texts provided for theme extraction")
        return np.array([]), []
    
    # Preprocess texts
    processed_texts = [str(text).lower() for text in texts if text and str(text).strip()]
    
    if len(processed_texts) == 0:
        logger.warning("No valid texts after preprocessing")
        return np.array([]), []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    try:
        # Vectorize texts
        tfidf = vectorizer.fit_transform(processed_texts)
        
        # Apply NMF
        nmf = NMF(
            n_components=min(n_topics, len(processed_texts)),
            random_state=42,
            max_iter=200
        )
        
        doc_topic = nmf.fit_transform(tfidf)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top terms for each topic
        topic_terms = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_terms = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topic_terms.append(", ".join(top_terms))
        
        logger.info(f"Extracted {len(topic_terms)} themes from {len(processed_texts)} texts")
        return doc_topic, topic_terms
    except Exception as e:
        logger.error(f"Error extracting themes: {str(e)}")
        return np.array([]), []

@timed
def extract_survey_themes(df: pd.DataFrame, n_topics: int = 5) -> pd.DataFrame:
    """
    Extract themes from survey responses.
    
    Args:
        df: DataFrame with survey data
        n_topics: Number of topics/themes to extract
        
    Returns:
        DataFrame with theme distributions for each response
    """
    if 'free_text' not in df.columns:
        logger.warning("No 'free_text' column found in survey data")
        return pd.DataFrame()
    
    # Get valid texts
    texts = df['free_text'].dropna().tolist()
    
    if len(texts) == 0:
        logger.warning("No valid texts found in survey data")
        return pd.DataFrame()
    
    # Extract themes
    doc_topic, topic_terms = extract_themes(texts, n_topics)
    
    if len(doc_topic) == 0:
        return pd.DataFrame()
    
    # Create result DataFrame
    result_df = df.copy()
    
    # Add theme columns
    for i in range(min(n_topics, doc_topic.shape[1])):
        result_df[f'theme_{i+1}_score'] = 0.0
    
    # Fill theme scores for valid texts
    valid_indices = df['free_text'].notna()
    for i in range(min(n_topics, doc_topic.shape[1])):
        result_df.loc[valid_indices, f'theme_{i+1}_score'] = doc_topic[:, i]
    
    # Log topic terms
    for i, terms in enumerate(topic_terms):
        logger.info(f"Topic {i+1}: {terms}")
    
    logger.info(f"Extracted themes for {len(result_df)} survey responses")
    return result_df

@timed
def get_theme_summary(df: pd.DataFrame, n_topics: int = 5) -> pd.DataFrame:
    """
    Get a summary of themes by team/practice.
    
    Args:
        df: DataFrame with survey data including theme scores
        n_topics: Number of topics to include in summary
        
    Returns:
        DataFrame with theme summary by team
    """
    # Check if theme columns exist
    theme_cols = [f'theme_{i+1}_score' for i in range(n_topics)]
    if not all(col in df.columns for col in theme_cols):
        logger.warning("Theme columns not found in DataFrame")
        return pd.DataFrame()
    
    # Load employee data to get team/practice information
    from app.storage import storage
    
    try:
        employees_df = storage.read_table('employees')
        df_with_team = df.merge(employees_df[['employee_id', 'practice']], on='employee_id', how='left')
        
        # Aggregate by practice (team)
        team_summary = []
        for team in df_with_team['practice'].unique():
            if pd.isna(team):
                continue
                
            team_data = df_with_team[df_with_team['practice'] == team]
            
            # Calculate average theme scores
            theme_scores = {}
            for col in theme_cols:
                theme_scores[col] = team_data[col].mean()
            
            # Find dominant theme
            dominant_theme_idx = int(max(theme_scores, key=theme_scores.get).split('_')[1]) - 1
            dominant_theme_score = theme_scores[f'theme_{dominant_theme_idx+1}_score']
            
            team_summary.append({
                'team': team,
                'response_count': len(team_data),
                'dominant_theme': f"Topic {dominant_theme_idx+1}",
                'dominant_theme_score': dominant_theme_score,
                **theme_scores
            })
        
        result_df = pd.DataFrame(team_summary)
        logger.info(f"Generated theme summary for {len(result_df)} teams")
        return result_df
    except Exception as e:
        logger.error(f"Error generating theme summary: {str(e)}")
        return pd.DataFrame()