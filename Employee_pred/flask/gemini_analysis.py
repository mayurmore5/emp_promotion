import google.generativeai as genai
import json
from datetime import datetime

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyD5s3ycLvW8XPE2YPbwf362993YCdOu3Ng"
genai.configure(api_key=GEMINI_API_KEY)

def analyze_employee_promotion(employee_data, prediction_result, probability):
    """
    Analyze employee data using Gemini AI and provide detailed insights and recommendations.
    
    Args:
        employee_data (dict): Employee input data
        prediction_result (str): Prediction result (eligible/not eligible)
        probability (float): Prediction probability
    
    Returns:
        dict: Detailed analysis with insights and recommendations
    """
    
    # Format employee data for analysis
    dept = employee_data.get('dept', 'Unknown')
    not_trainings = employee_data.get('not', 0)
    prev_rating = employee_data.get('pyr', 0)
    length_service = employee_data.get('los', 0)
    kpi_met = employee_data.get('kpi', 0)
    awards_won = employee_data.get('aw', 0)
    avg_training_score = employee_data.get('ats', 0)
    
    # Create detailed prompt for Gemini
    prompt = f"""
    You are an expert HR analyst and career development specialist. Analyze the following employee data and provide a comprehensive promotion analysis report.

    EMPLOYEE DATA:
    - Department: {dept}
    - Number of Trainings Completed: {not_trainings}
    - Previous Year Rating: {prev_rating}/5
    - Length of Service: {length_service} years
    - KPIs Met (>80%): {'Yes' if kpi_met else 'No'}
    - Awards Won: {'Yes' if awards_won else 'No'}
    - Average Training Score: {avg_training_score}/100

    PREDICTION RESULT:
    - Promotion Eligibility: {prediction_result}
    - Confidence Level: {probability:.1%}

    Please provide a detailed analysis including:

    1. CURRENT PERFORMANCE ASSESSMENT:
    - Strengths analysis
    - Areas of concern
    - Overall performance rating

    2. PROMOTION READINESS ANALYSIS:
    - Why the employee is/isn't ready for promotion
    - Key factors influencing the decision
    - Risk assessment

    3. SPECIFIC IMPROVEMENT RECOMMENDATIONS:
    - Short-term goals (3-6 months)
    - Medium-term development (6-12 months)
    - Long-term career path suggestions
    - Specific training recommendations
    - Skill development priorities

    4. ACTION PLAN:
    - Immediate next steps
    - Timeline for improvement
    - Success metrics to track
    - Mentorship/coaching suggestions

    5. COMPETITIVE ANALYSIS:
    - How they compare to peers
    - Industry benchmarks
    - Market positioning

    Format your response as a structured analysis with clear sections and actionable insights. Be specific, constructive, and provide measurable recommendations.
    """
    
    try:
        # Generate analysis using Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Parse and structure the response
        analysis_text = response.text
        
        # Extract key insights using additional prompts
        insights = extract_key_insights(employee_data, prediction_result, probability)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction_result': prediction_result,
            'confidence': probability,
            'detailed_analysis': analysis_text,
            'key_insights': insights,
            'employee_data': employee_data
        }
        
    except Exception as e:
        return {
            'timestamp': datetime.now().isoformat(),
            'error': f"Analysis failed: {str(e)}",
            'prediction_result': prediction_result,
            'confidence': probability,
            'employee_data': employee_data
        }

def extract_key_insights(employee_data, prediction_result, probability):
    """
    Extract key insights and metrics from the analysis.
    """
    
    dept = employee_data.get('dept', 'Unknown')
    not_trainings = employee_data.get('not', 0)
    prev_rating = employee_data.get('pyr', 0)
    length_service = employee_data.get('los', 0)
    kpi_met = employee_data.get('kpi', 0)
    awards_won = employee_data.get('aw', 0)
    avg_training_score = employee_data.get('ats', 0)
    
    insights = {
        'performance_score': calculate_performance_score(employee_data),
        'promotion_readiness': calculate_readiness_score(employee_data),
        'strengths': [],
        'weaknesses': [],
        'immediate_actions': [],
        'development_areas': []
    }
    
    # Analyze strengths
    if prev_rating >= 4.0:
        insights['strengths'].append("Strong performance rating")
    if kpi_met:
        insights['strengths'].append("Consistently meets KPIs")
    if awards_won:
        insights['strengths'].append("Recognition through awards")
    if avg_training_score >= 80:
        insights['strengths'].append("Excellent training performance")
    if not_trainings >= 5:
        insights['strengths'].append("High training engagement")
    
    # Analyze weaknesses
    if prev_rating < 3.5:
        insights['weaknesses'].append("Below-average performance rating")
    if not kpi_met:
        insights['weaknesses'].append("Struggles to meet KPIs")
    if avg_training_score < 70:
        insights['weaknesses'].append("Training performance needs improvement")
    if length_service < 2:
        insights['weaknesses'].append("Limited experience in role")
    
    # Generate immediate actions
    if prev_rating < 4.0:
        insights['immediate_actions'].append("Focus on improving performance metrics")
    if not kpi_met:
        insights['immediate_actions'].append("Develop action plan to meet KPIs")
    if avg_training_score < 80:
        insights['immediate_actions'].append("Enroll in additional training programs")
    
    # Development areas
    insights['development_areas'] = [
        "Leadership skills development",
        "Strategic thinking",
        "Cross-functional collaboration",
        "Industry knowledge expansion"
    ]
    
    return insights

def calculate_performance_score(employee_data):
    """Calculate overall performance score (0-100)."""
    score = 0
    
    # Previous year rating (40% weight)
    rating = employee_data.get('pyr', 0)
    score += (rating / 5) * 40
    
    # KPI achievement (25% weight)
    if employee_data.get('kpi', 0):
        score += 25
    
    # Training score (20% weight)
    training_score = employee_data.get('ats', 0)
    score += (training_score / 100) * 20
    
    # Awards (10% weight)
    if employee_data.get('aw', 0):
        score += 10
    
    # Training participation (5% weight)
    trainings = employee_data.get('not', 0)
    score += min(trainings * 2, 5)
    
    return round(score, 1)

def calculate_readiness_score(employee_data):
    """Calculate promotion readiness score (0-100)."""
    score = 0
    
    # Performance rating (30% weight)
    rating = employee_data.get('pyr', 0)
    score += (rating / 5) * 30
    
    # Experience (25% weight)
    service_years = employee_data.get('los', 0)
    score += min(service_years * 5, 25)
    
    # KPI achievement (20% weight)
    if employee_data.get('kpi', 0):
        score += 20
    
    # Training excellence (15% weight)
    training_score = employee_data.get('ats', 0)
    score += (training_score / 100) * 15
    
    # Recognition (10% weight)
    if employee_data.get('aw', 0):
        score += 10
    
    return round(score, 1)

def generate_improvement_plan(employee_data, prediction_result):
    """
    Generate a structured improvement plan for the employee.
    """
    
    plan = {
        'short_term': {
            'timeline': '3-6 months',
            'goals': [],
            'actions': []
        },
        'medium_term': {
            'timeline': '6-12 months',
            'goals': [],
            'actions': []
        },
        'long_term': {
            'timeline': '1-2 years',
            'goals': [],
            'actions': []
        }
    }
    
    # Short-term goals based on current data
    if employee_data.get('pyr', 0) < 4.0:
        plan['short_term']['goals'].append("Improve performance rating to 4.0+")
        plan['short_term']['actions'].append("Set up regular 1:1 meetings with manager")
        plan['short_term']['actions'].append("Create specific performance improvement plan")
    
    if not employee_data.get('kpi', 0):
        plan['short_term']['goals'].append("Achieve 80%+ KPI targets")
        plan['short_term']['actions'].append("Identify KPI gaps and create action plan")
        plan['short_term']['actions'].append("Request additional resources if needed")
    
    if employee_data.get('ats', 0) < 80:
        plan['short_term']['goals'].append("Improve training scores to 80+")
        plan['short_term']['actions'].append("Enroll in relevant training programs")
        plan['short_term']['actions'].append("Seek mentorship from high performers")
    
    # Medium-term goals
    plan['medium_term']['goals'].extend([
        "Develop leadership skills",
        "Expand cross-functional knowledge",
        "Build strategic thinking capabilities"
    ])
    
    plan['medium_term']['actions'].extend([
        "Take on stretch assignments",
        "Participate in cross-departmental projects",
        "Attend leadership development programs"
    ])
    
    # Long-term goals
    plan['long_term']['goals'].extend([
        "Prepare for next-level role",
        "Build industry expertise",
        "Develop mentoring capabilities"
    ])
    
    plan['long_term']['actions'].extend([
        "Shadow senior leaders",
        "Join industry associations",
        "Start mentoring junior employees"
    ])
    
    return plan
