# path: mindcare-backend/app/nlp/sentiment.py
import logging
import re
from typing import List, Union

import pandas as pd
from app.utils.timers import timed

logger = logging.getLogger(__name__)

# Try to import TextBlob, fallback to rule-based sentiment
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("Using TextBlob for sentiment analysis")
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available, falling back to rule-based sentiment")

# Simple sentiment lexicon for fallback
POSITIVE_WORDS = [
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 
    'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'appreciate',
    'helpful', 'supportive', 'collaborative', 'productive', 'efficient'
]

NEGATIVE_WORDS = [
    'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'disappointed',
    'frustrated', 'angry', 'upset', 'sad', 'unhappy', 'dissatisfied',
    'unhelpful', 'unsupportive', 'difficult', 'stressful', 'overwhelmed'
]

def rule_based_sentiment(text: str) -> float:
    """
    Calculate sentiment score using rule-based approach.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment score between -1 and 1
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Clean text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # Count positive and negative words
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    # Calculate sentiment score
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    sentiment = (positive_count - negative_count) / max(total_words, 1)
    
    # Clamp between -1 and 1
    return max(-1.0, min(1.0, sentiment))

def textblob_sentiment(text: str) -> float:
    """
    Calculate sentiment score using TextBlob.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment score between -1 and 1
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Clamp between -1 and 1
    return max(-1.0, min(1.0, polarity))

@timed
def analyze_sentiment(text: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Analyze sentiment of text or a list of texts.
    
    Args:
        text: Input text or list of texts
        
    Returns:
        Sentiment score or list of scores between -1 and 1
    """
    if TEXTBLOB_AVAILABLE:
        if isinstance(text, list):
            return [textblob_sentiment(t) for t in text]
        else:
            return textblob_sentiment(text)
    else:
        if isinstance(text, list):
            return [rule_based_sentiment(t) for t in text]
        else:
            return rule_based_sentiment(text)

@timed
def calculate_survey_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sentiment scores for survey responses.
    
    Args:
        df: DataFrame with survey data
        
    Returns:
        DataFrame with added sentiment_score column
    """
    df_result = df.copy()
    
    # Calculate sentiment for each survey response
    if 'free_text' in df_result.columns:
        df_result['sentiment_score'] = analyze_sentiment(df_result['free_text'].tolist())
    else:
        logger.warning("No 'free_text' column found in survey data")
        df_result['sentiment_score'] = 0.0
    
    # Normalize rating to -1 to 1 scale if available
    if 'rating' in df_result.columns:
        # Assuming rating is 1-5 scale
        min_rating, max_rating = 1, 5
        df_result['normalized_rating'] = 2 * (df_result['rating'] - min_rating) / (max_rating - min_rating) - 1
        
        # Combine sentiment and rating
        df_result['survey_sentiment'] = (df_result['sentiment_score'] + df_result['normalized_rating']) / 2
    else:
        df_result['survey_sentiment'] = df_result['sentiment_score']
    
    logger.info(f"Calculated sentiment for {len(df_result)} survey responses")
    return df_result

@timed
def aggregate_sentiment_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by team.
    
    Args:
        df: DataFrame with survey data including sentiment scores
        
    Returns:
        DataFrame with aggregated sentiment by team
    """
    if 'employee_id' not in df.columns or 'survey_sentiment' not in df.columns:
        logger.error("Required columns not found for sentiment aggregation")
        return pd.DataFrame()
    
    # Load employee data to get team/practice information
    from app.storage import storage
    
    try:
        employees_df = storage.read_table('employees')
        df_with_team = df.merge(employees_df[['employee_id', 'practice']], on='employee_id', how='left')
        
        # Aggregate by practice (team)
        team_sentiment = df_with_team.groupby('practice')['survey_sentiment'].agg(['mean', 'count']).reset_index()
        team_sentiment.columns = ['team', 'avg_sentiment', 'response_count']
        
        logger.info(f"Aggregated sentiment for {len(team_sentiment)} teams")
        return team_sentiment
    except Exception as e:
        logger.error(f"Error aggregating sentiment by team: {str(e)}")
        return pd.DataFrame()