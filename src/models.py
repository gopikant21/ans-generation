from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Domain(str, Enum):
    SOFTWARE_ENGINEERING = "Software Engineering"
    DATA_SCIENCE = "Data Science"
    MACHINE_LEARNING = "Machine Learning"
    HUMAN_RESOURCES = "Human Resources"
    GENERAL_KNOWLEDGE = "General Knowledge"
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"


class ReferenceApproach(str, Enum):
    CONCISE_DEFINITION = "concise_definition"
    DETAILED_EXPLANATION = "detailed_explanation"
    STEP_BY_STEP_SOLUTION = "step_by_step_solution"
    ANALOGY_OR_EXAMPLE = "analogy_or_example"
    FORMULA_BASED = "formula_based"


class CandidateType(str, Enum):
    PERFECT_MATCH = "perfect_match"
    ALTERNATE_CORRECT = "alternate_correct"
    PARTIAL_CORRECT = "partial_correct"
    MISCONCEPTION = "misconception"
    OFF_TOPIC = "off_topic"
    POOR_QUALITY = "poor_quality"


class QuestionInput(BaseModel):
    question: str = Field(..., description="The question to generate answers for")
    domain: Domain = Field(..., description="The domain/subject area")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)


class ReferenceAnswer(BaseModel):
    approach: ReferenceApproach
    answer: str = Field(..., description="The reference answer content")


class CandidateAnswer(BaseModel):
    type: CandidateType
    answer: str = Field(..., description="The candidate answer content")


class GeneratedAnswers(BaseModel):
    question: str
    domain: Domain
    difficulty: DifficultyLevel
    reference_answers: List[ReferenceAnswer] = Field(
        ..., description="3-5 different correct reference answers"
    )
    candidate_answers: List[CandidateAnswer] = Field(
        ..., description="6+ candidate answers with varying quality"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class BatchQuestionInput(BaseModel):
    questions: List[QuestionInput] = Field(
        ..., description="List of questions to process"
    )


class BatchGeneratedAnswers(BaseModel):
    results: List[GeneratedAnswers]
    total_questions: int
    successful_generations: int
    failed_generations: int


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
