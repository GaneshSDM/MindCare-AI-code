# path: mindcare-backend/app/recommender/career.py
import logging
from typing import Dict, List, Tuple

import pandas as pd
from app.storage import storage
from app.utils.timers import timed

logger = logging.getLogger(__name__)

# Role-skill requirements matrix
ROLE_SKILL_REQUIREMENTS = {
    'Data Engineer': {
        'Python': 3, 'SQL': 3, 'dbt': 2, 'Airflow': 2, 'Snowflake': 2,
        'Data Modeling': 2, 'ETL': 3, 'Cloud': 2
    },
    'ML Engineer': {
        'Python': 4, 'TensorFlow': 3, 'PyTorch': 3, 'Scikit-learn': 3, 'SQL': 2,
        'Statistics': 3, 'Machine Learning': 4, 'Deep Learning': 3
    },
    'Analyst': {
        'SQL': 3, 'Excel': 3, 'Tableau': 2, 'Python': 2, 'Statistics': 2,
        'Data Visualization': 2, 'Business Acumen': 2
    },
    'Manager': {
        'Leadership': 4, 'Communication': 3, 'Project Management': 3, 'Excel': 2, 'SQL': 1,
        'Team Building': 3, 'Strategic Thinking': 3
    },
    'Senior Data Engineer': {
        'Python': 4, 'SQL': 4, 'dbt': 3, 'Airflow': 3, 'Snowflake': 3,
        'Data Modeling': 3, 'ETL': 4, 'Cloud': 3, 'Architecture': 3
    },
    'Senior ML Engineer': {
        'Python': 4, 'TensorFlow': 4, 'PyTorch': 4, 'Scikit-learn': 4, 'SQL': 3,
        'Statistics': 4, 'Machine Learning': 4, 'Deep Learning': 4, 'MLOps': 3
    },
    'Senior Analyst': {
        'SQL': 4, 'Excel': 4, 'Tableau': 3, 'Python': 3, 'Statistics': 3,
        'Data Visualization': 3, 'Business Acumen': 3, 'Storytelling': 3
    }
}

# Career progression paths
CAREER_PATHS = {
    'Data Engineer': ['Senior Data Engineer', 'Manager'],
    'ML Engineer': ['Senior ML Engineer', 'Manager'],
    'Analyst': ['Senior Analyst', 'Data Engineer', 'ML Engineer'],
    'Senior Data Engineer': ['Manager'],
    'Senior ML Engineer': ['Manager'],
    'Senior Analyst': ['Manager']
}

# Project rotation options
PROJECT_ROTATIONS = [
    {'title': 'Data Migration Project', 'skills': ['SQL', 'ETL', 'Cloud'], 'duration': '3 months'},
    {'title': 'ML Model Deployment', 'skills': ['Python', 'MLOps', 'Cloud'], 'duration': '4 months'},
    {'title': 'Data Visualization Dashboard', 'skills': ['Tableau', 'SQL', 'Python'], 'duration': '2 months'},
    {'title': 'Data Quality Initiative', 'skills': ['Python', 'SQL', 'Data Modeling'], 'duration': '3 months'},
    {'title': 'Customer Analytics Project', 'skills': ['SQL', 'Statistics', 'Business Acumen'], 'duration': '3 months'}
]

@timed
def get_employee_skills(employee_id: str) -> Dict[str, int]:
    """
    Get skills for an employee.
    
    Args:
        employee_id: Employee ID
        
    Returns:
        Dictionary of skills and levels
    """
    try:
        # Load skills data
        skills_df = storage.read_table('skills')
        
        # Filter for employee
        employee_skills = skills_df[skills_df['employee_id'] == employee_id]
        
        # Convert to dictionary
        skills_dict = {}
        for _, row in employee_skills.iterrows():
            skills_dict[row['skill']] = row['level']
        
        return skills_dict
    except Exception as e:
        logger.error(f"Error getting employee skills: {str(e)}")
        return {}

@timed
def get_employee_info(employee_id: str) -> Dict:
    """
    Get information for an employee.
    
    Args:
        employee_id: Employee ID
        
    Returns:
        Dictionary with employee information
    """
    try:
        # Load employee data
        employees_df = storage.read_table('employees')
        
        # Filter for employee
        employee_data = employees_df[employees_df['employee_id'] == employee_id]
        
        if len(employee_data) == 0:
            logger.warning(f"Employee {employee_id} not found")
            return {}
        
        # Convert to dictionary
        employee_info = employee_data.iloc[0].to_dict()
        
        return employee_info
    except Exception as e:
        logger.error(f"Error getting employee info: {str(e)}")
        return {}

@timed
def calculate_skill_gap(employee_skills: Dict[str, int], target_role: str) -> Tuple[float, Dict[str, int]]:
    """
    Calculate skill gap for an employee to reach a target role.
    
    Args:
        employee_skills: Dictionary of employee skills and levels
        target_role: Target role
        
    Returns:
        Tuple of (skill gap score, skill gaps)
    """
    try:
        # Get target role requirements
        if target_role not in ROLE_SKILL_REQUIREMENTS:
            logger.warning(f"Unknown target role: {target_role}")
            return 1.0, {}
        
        required_skills = ROLE_SKILL_REQUIREMENTS[target_role]
        
        # Calculate skill gaps
        skill_gaps = {}
        total_gap = 0
        total_required = 0
        
        for skill, required_level in required_skills.items():
            actual_level = employee_skills.get(skill, 0)
            gap = max(0, required_level - actual_level)
            skill_gaps[skill] = gap
            
            total_gap += gap
            total_required += required_level
        
        # Calculate normalized gap score
        gap_score = total_gap / total_required if total_required > 0 else 1.0
        
        return gap_score, skill_gaps
    except Exception as e:
        logger.error(f"Error calculating skill gap: {str(e)}")
        return 1.0, {}

@timed
def recommend_courses(employee_skills: Dict[str, int], skill_gaps: Dict[str, int], limit: int = 4) -> List[Dict]:
    """
    Recommend courses for an employee based on skill gaps.
    
    Args:
        employee_skills: Dictionary of employee skills and levels
        skill_gaps: Dictionary of skill gaps
        limit: Maximum number of courses to recommend
        
    Returns:
        List of recommended courses
    """
    try:
        # Load learning catalog
        try:
            ld_catalog_df = storage.read_table('ld_catalog')
        except:
            # Create a mock catalog if not available
            logger.warning("Learning catalog not available, using mock data")
            ld_catalog_df = pd.DataFrame([
                {'course_id': 'C101', 'title': 'Python Advanced Programming', 'skill': 'Python', 'provider': 'Internal'},
                {'course_id': 'C102', 'title': 'SQL for Data Analysis', 'skill': 'SQL', 'provider': 'Internal'},
                {'course_id': 'C103', 'title': 'Machine Learning Fundamentals', 'skill': 'Machine Learning', 'provider': 'External'},
                {'course_id': 'C104', 'title': 'Data Visualization with Tableau', 'skill': 'Tableau', 'provider': 'Internal'},
                {'course_id': 'C105', 'title': 'Leadership Training', 'skill': 'Leadership', 'provider': 'External'},
                {'course_id': 'C106', 'title': 'Cloud Computing Fundamentals', 'skill': 'Cloud', 'provider': 'External'},
                {'course_id': 'C107', 'title': 'ETL Best Practices', 'skill': 'ETL', 'provider': 'Internal'},
                {'course_id': 'C108', 'title': 'Statistical Analysis', 'skill': 'Statistics', 'provider': 'External'},
                {'course_id': 'C109', 'title': 'Project Management', 'skill': 'Project Management', 'provider': 'External'},
                {'course_id': 'C110', 'title': 'Communication Skills', 'skill': 'Communication', 'provider': 'Internal'}
            ])
        
        # Sort skills by gap size
        sorted_skills = sorted(skill_gaps.items(), key=lambda x: x[1], reverse=True)
        
        # Get top skills with gaps
        top_skills = [skill for skill, gap in sorted_skills if gap > 0]
        
        # Find courses for top skills
        recommended_courses = []
        
        for skill in top_skills:
            if len(recommended_courses) >= limit:
                break
            
            # Find courses for this skill
            skill_courses = ld_catalog_df[ld_catalog_df['skill'] == skill]
            
            if len(skill_courses) > 0:
                # Get the first course for this skill
                course = skill_courses.iloc[0].to_dict()
                recommended_courses.append({
                    'course_id': course['course_id'],
                    'title': course['title'],
                    'skill': course['skill'],
                    'provider': course['provider'],
                    'reason': f"Address skill gap in {skill}"
                })
        
        # If we still need more courses, add general ones
        if len(recommended_courses) < limit:
            remaining = limit - len(recommended_courses)
            general_courses = ld_catalog_df.head(remaining)
            
            for _, course in general_courses.iterrows():
                recommended_courses.append({
                    'course_id': course['course_id'],
                    'title': course['title'],
                    'skill': course['skill'],
                    'provider': course['provider'],
                    'reason': "General professional development"
                })
        
        return recommended_courses[:limit]
    except Exception as e:
        logger.error(f"Error recommending courses: {str(e)}")
        return []

@timed
def recommend_mentor(employee_id: str, employee_info: Dict) -> Dict:
    """
    Recommend a mentor for an employee.
    
    Args:
        employee_id: Employee ID
        employee_info: Employee information
        
    Returns:
        Dictionary with mentor recommendation
    """
    try:
        # Load employee data
        employees_df = storage.read_table('employees')
        
        # Get employee's practice and seniority
        practice = employee_info.get('practice', '')
        seniority = employee_info.get('seniority', '')
        
        # Define seniority hierarchy
        seniority_hierarchy = {
            'Junior': 0,
            'Mid': 1,
            'Senior': 2,
            'Lead': 3,
            'Manager': 4
        }
        
        # Find potential mentors
        potential_mentors = []
        
        for _, emp in employees_df.iterrows():
            # Skip self
            if emp['employee_id'] == employee_id:
                continue
            
            # Check if same practice
            if emp['practice'] != practice:
                continue
            
            # Check if higher seniority
            emp_seniority = emp.get('seniority', '')
            if (seniority_hierarchy.get(emp_seniority, 0) <= seniority_hierarchy.get(seniority, 0)):
                continue
            
            potential_mentors.append(emp.to_dict())
        
        # If no mentors found in same practice, look in any practice
        if not potential_mentors:
            for _, emp in employees_df.iterrows():
                # Skip self
                if emp['employee_id'] == employee_id:
                    continue
                
                # Check if higher seniority
                emp_seniority = emp.get('seniority', '')
                if (seniority_hierarchy.get(emp_seniority, 0) <= seniority_hierarchy.get(seniority, 0)):
                    continue
                
                potential_mentors.append(emp.to_dict())
        
        # If still no mentors found, return empty
        if not potential_mentors:
            return {
                'mentor_id': None,
                'mentor_name': None,
                'mentor_role': None,
                'mentor_practice': None,
                'reason': 'No suitable mentors found'
            }
        
        # Select a mentor (for demo, we'll just pick the first one)
        mentor = potential_mentors[0]
        
        return {
            'mentor_id': mentor['employee_id'],
            'mentor_name': f"Mentor {mentor['employee_id']}",  # In a real system, we'd have actual names
            'mentor_role': mentor['role'],
            'mentor_practice': mentor['practice'],
            'reason': f"Senior {mentor['role']} in {mentor['practice']} practice"
        }
    except Exception as e:
        logger.error(f"Error recommending mentor: {str(e)}")
        return {
            'mentor_id': None,
            'mentor_name': None,
            'mentor_role': None,
            'mentor_practice': None,
            'reason': f'Error: {str(e)}'
        }

@timed
def recommend_project_rotation(employee_skills: Dict[str, int], skill_gaps: Dict[str, int]) -> Dict:
    """
    Recommend a project rotation for an employee.
    
    Args:
        employee_skills: Dictionary of employee skills and levels
        skill_gaps: Dictionary of skill gaps
        
    Returns:
        Dictionary with project rotation recommendation
    """
    try:
        # Get top skills with gaps
        sorted_skills = sorted(skill_gaps.items(), key=lambda x: x[1], reverse=True)
        top_skills = [skill for skill, gap in sorted_skills if gap > 0][:2]  # Top 2 skills
        
        # Find projects that match top skills
        matching_projects = []
        
        for project in PROJECT_ROTATIONS:
            project_skills = project.get('skills', [])
            
            # Count how many top skills this project addresses
            match_count = sum(1 for skill in top_skills if skill in project_skills)
            
            if match_count > 0:
                matching_projects.append({
                    **project,
                    'match_count': match_count
                })
        
        # Sort by match count
        matching_projects.sort(key=lambda x: x['match_count'], reverse=True)
        
        # Select the best project
        if matching_projects:
            project = matching_projects[0]
            return {
                'title': project['title'],
                'duration': project['duration'],
                'skills': project['skills'],
                'reason': f"Addresses skill gaps in {', '.join(top_skills)}"
            }
        else:
            # If no matching projects, return a random one
            project = PROJECT_ROTATIONS[0]
            return {
                'title': project['title'],
                'duration': project['duration'],
                'skills': project['skills'],
                'reason': "General skill development"
            }
    except Exception as e:
        logger.error(f"Error recommending project rotation: {str(e)}")
        return {
            'title': None,
            'duration': None,
            'skills': [],
            'reason': f'Error: {str(e)}'
        }

@timed
def generate_career_plan(employee_id: str) -> Dict:
    """
    Generate a career plan for an employee.
    
    Args:
        employee_id: Employee ID
        
    Returns:
        Dictionary with career plan
    """
    try:
        # Get employee information
        employee_info = get_employee_info(employee_id)
        
        if not employee_info:
            return {
                'employee_id': employee_id,
                'error': 'Employee not found'
            }
        
        # Get employee skills
        employee_skills = get_employee_skills(employee_id)
        
        # Get current role
        current_role = employee_info.get('role', '')
        
        # Get potential next roles
        next_roles = CAREER_PATHS.get(current_role, [current_role])
        
        # Calculate skill gaps for each next role
        role_gaps = {}
        for role in next_roles:
            gap_score, skill_gaps = calculate_skill_gap(employee_skills, role)
            role_gaps[role] = {
                'gap_score': gap_score,
                'skill_gaps': skill_gaps
            }
        
        # Select the role with the smallest gap (or the first one if all are equal)
        best_role = min(role_gaps.items(), key=lambda x: x[1]['gap_score'])[0]
        best_gap_info = role_gaps[best_role]
        
        # Recommend courses
        courses = recommend_courses(employee_skills, best_gap_info['skill_gaps'])
        
        # Recommend mentor
        mentor = recommend_mentor(employee_id, employee_info)
        
        # Recommend project rotation
        project_rotation = recommend_project_rotation(employee_skills, best_gap_info['skill_gaps'])
        
        # Create career plan
        career_plan = {
            'employee_id': employee_id,
            'current_role': current_role,
            'target_role': best_role,
            'skill_gap_score': best_gap_info['gap_score'],
            'skill_gaps': best_gap_info['skill_gaps'],
            'recommended_courses': courses,
            'recommended_mentor': mentor,
            'recommended_project_rotation': project_rotation
        }
        
        logger.info(f"Generated career plan for employee {employee_id}")
        return career_plan
    except Exception as e:
        logger.error(f"Error generating career plan: {str(e)}")
        return {
            'employee_id': employee_id,
            'error': str(e)
        }
    