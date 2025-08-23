# path: mindcare-backend/app/routers/features.py
import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.features.build import build_all_features
from app.storage import storage

logger = logging.getLogger(__name__)
router = APIRouter()

class FeatureResponse(BaseModel):
    success: bool
    message: str
    tables_created: list = []

@router.post("/features/build", response_model=FeatureResponse)
async def build_features():
    """
    Build features from raw data.
    
    Returns:
        FeatureResponse with results
    """
    try:
        # Check if raw data tables exist
        required_tables = ['employees', 'surveys', 'timesheets', 'skills']
        missing_tables = [table for table in required_tables if not storage.table_exists(table)]
        
        if missing_tables:
            raise HTTPException(
                status_code=400, 
                detail=f"Required tables not found: {', '.join(missing_tables)}. Run ingestion first."
            )
        
        # Build features
        success = build_all_features()
        
        if not success:
            raise HTTPException(status_code=500, detail="Feature building failed")
        
        # Check if feature tables were created
        feature_tables = ['derived_employee_features', 'derived_team_risk']
        created_tables = [table for table in feature_tables if storage.table_exists(table)]
        
        return FeatureResponse(
            success=True,
            message=f"Built features and created {len(created_tables)} feature tables",
            tables_created=created_tables
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building features: {str(e)}")