# path: mindcare-backend/app/routers/risk.py
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.storage import storage

logger = logging.getLogger(__name__)
router = APIRouter()

class TeamRiskResponse(BaseModel):
    team: str
    avg_risk: float
    top_driver_1: str
    top_driver_2: str
    top_driver_3: str

@router.get("/risk/teams", response_model=List[TeamRiskResponse])
async def get_team_risk():
    """
    Get aggregated risk by team/practice.
    
    Returns:
        List of TeamRiskResponse with team risk data
    """
    try:
        # Check if team risk table exists
        if not storage.table_exists('derived_team_risk'):
            raise HTTPException(
                status_code=400, 
                detail="Team risk table does not exist. Run feature building first."
            )
        
        # Load team risk data
        team_risk_df = storage.read_table('derived_team_risk')
        
        if team_risk_df.empty:
            raise HTTPException(status_code=404, detail="No team risk data found")
        
        # Convert to list of dictionaries
        team_risk_list = team_risk_df.to_dict('records')
        
        # Ensure all required fields are present
        result = []
        for item in team_risk_list:
            result.append(TeamRiskResponse(
                team=item.get('team', 'Unknown'),
                avg_risk=float(item.get('avg_risk', 0.0)),
                top_driver_1=item.get('top_driver_1', 'Unknown'),
                top_driver_2=item.get('top_driver_2', 'Unknown'),
                top_driver_3=item.get('top_driver_3', 'Unknown')
            ))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting team risk: {str(e)}")

@router.get("/export/team-risk.csv")
async def export_team_risk():
    """
    Export team risk data as CSV.
    
    Returns:
        CSV file with team risk data
    """
    try:
        # Check if team risk table exists
        if not storage.table_exists('derived_team_risk'):
            raise HTTPException(
                status_code=400, 
                detail="Team risk table does not exist. Run feature building first."
            )
        
        # Load team risk data
        team_risk_df = storage.read_table('derived_team_risk')
        
        if team_risk_df.empty:
            raise HTTPException(status_code=404, detail="No team risk data found")
        
        # Convert to CSV
        csv_data = team_risk_df.to_csv(index=False)
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=team-risk.csv"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting team risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting team risk: {str(e)}")