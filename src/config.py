"""
Configuration settings for the Answer Generation System.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the application."""

    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

    # LangSmith Configuration
    LANGCHAIN_TRACING_V2: bool = (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "answer-generation")

    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-2.5-pro")
    REFERENCE_MODEL: str = os.getenv("REFERENCE_MODEL", "gemini-2.5-pro")
    CANDIDATE_MODEL: str = os.getenv(
        "CANDIDATE_MODEL", "gemini-2.5-pro"
    )  # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Generation Configuration
    DEFAULT_TEMPERATURE: float = 0.7
    MAX_RETRIES: int = 3
    MAX_BATCH_SIZE: int = 50
    DEFAULT_MAX_WORKERS: int = 3

    # Output Configuration
    DEFAULT_OUTPUT_DIR: str = "output"

    @classmethod
    def get_model_config(cls, model_type: str = "default") -> Dict[str, Any]:
        """
        Get model configuration based on type.

        Args:
            model_type: Type of model ("default", "reference", "candidate")

        Returns:
            Dictionary with model configuration
        """
        model_mapping = {
            "default": cls.DEFAULT_MODEL,
            "reference": cls.REFERENCE_MODEL,
            "candidate": cls.CANDIDATE_MODEL,
        }

        model_name = model_mapping.get(model_type, cls.DEFAULT_MODEL)

        return {
            "model_name": model_name,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_retries": cls.MAX_RETRIES,
        }

    @classmethod
    def validate_configuration(cls) -> Dict[str, bool]:
        """
        Validate that required configuration is present.

        Returns:
            Dictionary with validation results
        """
        validations = {}

        # Check API keys
        validations["google_key"] = bool(cls.GOOGLE_API_KEY)
        validations["has_llm_key"] = validations["google_key"]

        # Check optional configurations
        validations["langsmith_configured"] = (
            bool(cls.LANGCHAIN_API_KEY) and cls.LANGCHAIN_TRACING_V2
        )

        return validations

    @classmethod
    def get_llm_provider(cls, model_name: str = None) -> str:
        """
        Determine LLM provider based on model name.

        Args:
            model_name: Name of the model

        Returns:
            Provider name ("google", "unknown")
        """
        if model_name is None:
            model_name = cls.DEFAULT_MODEL

        model_name_lower = model_name.lower()

        if "gemini" in model_name_lower or "google" in model_name_lower:
            return "google"
        else:
            return "unknown"


# Domain-specific configuration
DOMAIN_CONFIG = {
    "Software Engineering": {
        "common_topics": [
            "programming paradigms",
            "software design patterns",
            "testing methodologies",
            "version control",
            "databases",
            "web development",
            "mobile development",
        ],
        "difficulty_modifiers": {
            "beginner": "Focus on basic concepts and definitions",
            "intermediate": "Include practical examples and applications",
            "advanced": "Cover complex scenarios and trade-offs",
        },
    },
    "Data Science": {
        "common_topics": [
            "statistics",
            "machine learning",
            "data visualization",
            "data preprocessing",
            "feature engineering",
            "model evaluation",
            "big data",
        ],
        "difficulty_modifiers": {
            "beginner": "Emphasize conceptual understanding",
            "intermediate": "Include mathematical foundations and tools",
            "advanced": "Cover advanced algorithms and research topics",
        },
    },
    "Machine Learning": {
        "common_topics": [
            "supervised learning",
            "unsupervised learning",
            "deep learning",
            "neural networks",
            "optimization",
            "regularization",
            "ensemble methods",
        ],
        "difficulty_modifiers": {
            "beginner": "Focus on intuitive explanations",
            "intermediate": "Include algorithm details and implementation",
            "advanced": "Cover theoretical foundations and cutting-edge research",
        },
    },
    "Human Resources": {
        "common_topics": [
            "recruitment",
            "performance management",
            "employee relations",
            "compensation",
            "training and development",
            "employment law",
            "organizational behavior",
        ],
        "difficulty_modifiers": {
            "beginner": "Cover basic HR principles",
            "intermediate": "Include practical HR scenarios",
            "advanced": "Address complex legal and strategic issues",
        },
    },
}

# Answer quality configuration
ANSWER_QUALITY_CONFIG = {
    "reference_answers": {
        "min_length": 50,  # Minimum characters
        "max_length": 1000,  # Maximum characters
        "required_approaches": [
            "concise_definition",
            "detailed_explanation",
            "step_by_step_solution",
            "analogy_or_example",
        ],
        "optional_approaches": ["formula_based"],
    },
    "candidate_answers": {
        "required_types": [
            "perfect_match",
            "alternate_correct",
            "partial_correct",
            "misconception",
            "off_topic",
            "poor_quality",
        ],
        "score_thresholds": {
            "perfect_match": 0.95,
            "alternate_correct": 0.85,
            "partial_correct": 0.65,
            "misconception": 0.25,
            "off_topic": 0.15,
            "poor_quality": 0.35,
        },
    },
}
