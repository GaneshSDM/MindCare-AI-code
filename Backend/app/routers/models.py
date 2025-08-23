# path: mindcare-backend/app/routers/model.py
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.attrition import predict_attrition, train_model
from app.models.explain import explain_prediction
from app.storage import storage

logger = logging.getLogger(__name__)
router = APIRouter()

class TrainModelRequest(BaseModel):
    model_type: str = Field(default="logistic_regression", description="Type of model to train")

class TrainModelResponse(BaseModel):
    success: bool
    message: str
    model_type: str

class PredictAttritionRequest(BaseModel):
    model_type: str = Field(default="logistic_regression", description="Type of model to use")
    employee_ids: Optional[List[str]] = Field(default=None, description="List of employee IDs to predict for")

class PredictAttritionResponse(BaseModel):
    success: bool
    message: str
    predictions: List[Dict] = []

class ExplainRequest(BaseModel):
    model_type: str = Field(default="logistic_regression", description="Type of model to use")
    method: str = Field(default="auto", description="Method to use for explanation")

class ExplainResponse(BaseModel):
    success: bool
    message: str
    explanation: Dict = {}

@router.post("/model/train", response_model=TrainModelResponse)
async def train_attrition_model(request: TrainModelRequest):
    """
    Train an attrition model.
    
    Args:
        request: TrainModelRequest with model type
        
    Returns:
        TrainModelResponse with results
    """
    try:
        # Check if feature table exists
        if not storage.table_exists('derived_employee_features'):
            raise HTTPException(
                status_code=400, 
                detail="Feature table does not exist. Run feature building first."
            )
        
        # Train model
        success, message = train_model(request.model_type)
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return TrainModelResponse(
            success=True,
            message=message,
            model_type=request.model_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.post("/predict/attrition", response_model=PredictAttritionResponse)
async def predict_attrition_risk(request: PredictAttritionRequest):
    """
    Predict attrition risk for employees.
    
    Args:
        request: PredictAttritionRequest with model type and employee IDs
        
    Returns:
        PredictAttritionResponse with predictions
    """
    try:
        # Check if feature table exists
        if not storage.table_exists('derived_employee_features'):
            raise HTTPException(
                status_code=400, 
                detail="Feature table does not exist. Run feature building first."
            )
        
        # Get predictions
        predictions_df = predict_attrition(request.model_type, request.employee_ids)
        
        if predictions_df.empty:
            raise HTTPException(status_code=404, detail="No employees found for prediction")
        
        # Convert to list of dictionaries
        predictions = predictions_df.to_dict('records')
        
        return PredictAttritionResponse(
            success=True,
            message=f"Generated predictions for {len(predictions)} employees",
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting attrition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting attrition: {str(e)}")

@router.get("/explain/{employee_id}", response_model=ExplainResponse)
async def explain_employee_attrition(
    employee_id: str,
    model_type: str = Query(default="logistic_regression", description="Type of model to use"),
    method: str = Query(default="auto", description="Method to use for explanation")
):
    """
    Explain attrition prediction for an employee.
    
    Args:
        employee_id: Employee ID
        model_type: Type of model to use
        method: Method to use for explanation
        
    Returns:
        ExplainResponse with explanation
    """
    try:
        # Check if feature table exists
        if not storage.table_exists('derived_employee_features'):
            raise HTTPException(
                status_code=400, 
                detail="Feature table does not exist. Run feature building first."
            )
        
        # Load features
        features_df = storage.read_table('derived_employee_features')
        
        # Check if employee exists
        if employee_id not in features_df['employee_id'].values:
            raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
        
        # Import model class
        from app.models.attrition import AttritionModel
        
        # Initialize model
        model = AttritionModel(model_type)
        
        # Explain prediction
        explanation = explain_prediction(model, features_df, employee_id, method)
        
        if not explanation:
            raise HTTPException(status_code=500, detail="Failed to generate explanation")
        
        return ExplainResponse(
            success=True,
            message=f"Generated explanation for employee {employee_id}",
            explanation=explanation
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error explaining prediction: {str(e)}")