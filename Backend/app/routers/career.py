# path: mindcare-backend/app/routers/career.py
import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from app.recommender.career import generate_career_plan

logger = logging.getLogger(__name__)
router = APIRouter()

class CourseRecommendation(BaseModel):
    course_id: str
    title: str
    skill: str
    provider: str
    reason: str

class MentorRecommendation(BaseModel):
    mentor_id: str
    mentor_name: str
    mentor_role: str
    mentor_practice: str
    reason: str

class ProjectRotationRecommendation(BaseModel):
    title: str
    duration: str
    skills: list
    reason: str

class CareerPlanResponse(BaseModel):
    employee_id: str
    current_role: str
    target_role: str
    skill_gap_score: float
    skill_gaps: Dict[str, int]
    recommended_courses: list[CourseRecommendation]
    recommended_mentor: MentorRecommendation
    recommended_project_rotation: ProjectRotationRecommendation

@router.get("/employee/{employee_id}/career-plan", response_model=CareerPlanResponse)
async def get_career_plan(employee_id: str = Path(..., description="Employee ID")):
    """
    Get a career plan for an employee.
    
    Args:
        employee_id: Employee ID
        
    Returns:
        CareerPlanResponse with career plan
    """
    try:
        # Generate career plan
        career_plan = generate_career_plan(employee_id)
        
        if 'error' in career_plan:
            raise HTTPException(status_code=404, detail=career_plan['error'])
        
        # Format response
        response = CareerPlanResponse(
            employee_id=career_plan['employee_id'],
            current_role=career_plan['current_role'],
            target_role=career_plan['target_role'],
            skill_gap_score=career_plan['skill_gap_score'],
            skill_gaps=career_plan['skill_gaps'],
            recommended_courses=[
                CourseRecommendation(**course) for course in career_plan['recommended_courses']
            ],
            recommended_mentor=MentorRecommendation(**career_plan['recommended_mentor']),
            recommended_project_rotation=ProjectRotationRecommendation(**career_plan['recommended_project_rotation'])
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating career plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating career plan: {str(e)}")