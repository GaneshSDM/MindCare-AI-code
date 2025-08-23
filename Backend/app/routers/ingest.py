# path: mindcare-backend/app/routers/ingest.py
import logging
import os
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import config
from app.storage import storage
from app.utils.pii import mask_columns
from app.utils.timers import timed

logger = logging.getLogger(__name__)
router = APIRouter()

class IngestResponse(BaseModel):
    success: bool
    message: str
    tables_created: List[str] = []

@timed
def run_ingest() -> Dict:
    """
    Run the data ingestion process.
    
    Returns:
        Dictionary with ingestion results
    """
    try:
        # Get data directory
        data_dir = Path(config.DATA_DIR)
        
        if not data_dir.exists():
            return {
                "success": False,
                "message": f"Data directory not found: {data_dir}",
                "tables_created": []
            }
        
        # Define CSV files and their corresponding table names
        csv_files = {
            "employees.csv": "employees",
            "surveys.csv": "surveys",
            "timesheets.csv": "timesheets",
            "skills.csv": "skills",
            "ld_catalog.csv": "ld_catalog"
        }
        
        tables_created = []
        
        # Process each CSV file
        for csv_file, table_name in csv_files.items():
            csv_path = data_dir / csv_file
            
            if csv_path.exists():
                try:
                    # Read CSV
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # Apply PII masking
                    pii_columns = [col for col in df.columns if 'id' in col.lower() or 'name' in col.lower() or 'email' in col.lower()]
                    if pii_columns:
                        df = mask_columns(df, pii_columns)
                    
                    # Create table
                    if storage.DUCKDB_AVAILABLE:
                        # DuckDB can directly create a table from a DataFrame
                        storage.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
                    else:
                        # SQLite requires creating the table first
                        df.to_sql(table_name, storage.conn, if_exists='replace', index=False)
                    
                    tables_created.append(table_name)
                    logger.info(f"Created table {table_name} from {csv_file}")
                except Exception as e:
                    logger.error(f"Error processing {csv_file}: {str(e)}")
            else:
                logger.warning(f"CSV file not found: {csv_path}")
        
        return {
            "success": len(tables_created) > 0,
            "message": f"Ingested {len(tables_created)} tables",
            "tables_created": tables_created
        }
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        return {
            "success": False,
            "message": f"Error during ingestion: {str(e)}",
            "tables_created": []
        }

@router.post("/ingest/run", response_model=IngestResponse)
async def ingest_data():
    """
    Ingest data from CSV files into the database.
    
    Returns:
        IngestResponse with results
    """
    result = run_ingest()
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    return IngestResponse(
        success=result["success"],
        message=result["message"],
        tables_created=result["tables_created"]
    )