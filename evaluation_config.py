# Evaluation Configuration
EVALUATION_CONFIG = {
    # OpenAI Configuration
    "openai_model": "gpt-4o-mini",  # or "gpt-4o" for better quality
    "temperature": 0.0,  # Low temperature for consistent evaluation
    # Evaluation Weights
    "scoring_weights": {
        "factual_accuracy": 0.35,
        "relevance": 0.45,
        "completeness": 0.20,
    },
    # Scoring Thresholds
    "score_thresholds": {
        "excellent": 90,
        "good": 80,
        "satisfactory": 70,
        "below_average": 60,
    },
    # File Paths
    "input_file": "response_1755348961176.json",
    "output_directory": "evaluation_results",
    # Evaluation Settings
    "max_retries": 3,
    "batch_size": 5,  # Number of answers to evaluate before saving progress
    "save_progress": True,  # Save intermediate results
    # Report Settings
    "generate_excel": True,
    "generate_plots": True,
    "include_detailed_feedback": True,
}

# Prompt Templates
EVALUATION_PROMPTS = {
    "main_template": """
You are an expert evaluator for interview responses. Compare the STUDENT ANSWER to the REFERENCE ANSWERS if available, or evaluate based on your expertise.

Question: {question}
Domain: {domain}
Difficulty: {difficulty}

Reference Answers:
{references}

Student Answer:
{student_answer}

Evaluation Criteria:
- RELEVANCE (0-100): How well does the answer address the specific question asked?
- FACTUAL_ACCURACY (0-100): How technically correct and accurate is the information provided?
- COMPLETENESS (0-100): How comprehensive is the answer in covering the important aspects?

Scoring Guidelines:
- 90-100: Excellent - Comprehensive, accurate, and highly relevant
- 80-89: Good - Mostly accurate and relevant with minor gaps
- 70-79: Satisfactory - Generally correct but missing some important points
- 60-69: Below Average - Some correct information but significant gaps
- 0-59: Poor - Major inaccuracies or irrelevant content

Additional Context:
- Consider the domain and difficulty level when evaluating
- For technical questions, prioritize accuracy and depth
- For business questions, prioritize practical relevance and clarity
- For behavioral questions, look for structure and specific examples

Formula: overall_score = {factual_weight} * factual_accuracy + {relevance_weight} * relevance + {completeness_weight} * completeness

Respond ONLY with valid JSON in this format:
{{
  "relevance": <float>,
  "factual_accuracy": <float>,
  "completeness": <float>,
  "overall_score": <float>,
  "strengths": "<brief description of what was done well>",
  "weaknesses": "<brief description of areas for improvement>",
  "grade": "<letter grade A-F>"
}}
""",
    "feedback_template": """
Based on the evaluation scores:
- Overall Score: {overall_score:.1f}/100
- Relevance: {relevance:.1f}/100
- Factual Accuracy: {factual_accuracy:.1f}/100
- Completeness: {completeness:.1f}/100

Strengths: {strengths}
Areas for Improvement: {weaknesses}
Grade: {grade}
""",
}
