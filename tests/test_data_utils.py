import os
import tempfile
import json
from src.utils.data_utils import (
    export_to_json,
    export_to_csv,
    load_questions_from_json,
    validate_question_format,
    create_evaluation_dataset,
)
from src.models import (
    GeneratedAnswers,
    Domain,
    DifficultyLevel,
    ReferenceAnswer,
    CandidateAnswer,
)


class TestDataUtils:
    """Test cases for data utility functions."""

    def test_export_to_json(self):
        """Test JSON export functionality."""
        # Create test data
        answers = GeneratedAnswers(
            question="What is AI?",
            domain=Domain.DATA_SCIENCE,
            difficulty=DifficultyLevel.BEGINNER,
            reference_answers=[
                ReferenceAnswer(
                    approach="concise_definition", answer="AI is intelligent machines"
                )
            ],
            candidate_answers=[
                CandidateAnswer(
                    type="perfect_match", answer="AI means artificial intelligence"
                )
            ],
            metadata={"test": True},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            result_path = export_to_json(answers, filepath)
            assert result_path == filepath
            assert os.path.exists(filepath)

            # Verify content
            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["question"] == "What is AI?"
            assert data["domain"] == "Data Science"
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_export_to_csv(self):
        """Test CSV export functionality."""
        answers = GeneratedAnswers(
            question="What is ML?",
            domain=Domain.MACHINE_LEARNING,
            difficulty=DifficultyLevel.INTERMEDIATE,
            reference_answers=[
                ReferenceAnswer(
                    approach="detailed_explanation",
                    answer="Machine learning explanation",
                )
            ],
            candidate_answers=[
                CandidateAnswer(type="misconception", answer="ML is magic")
            ],
            metadata={},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        try:
            result_path = export_to_csv(answers, filepath)
            assert result_path == filepath
            assert os.path.exists(filepath)

            # Verify content
            with open(filepath, "r") as f:
                content = f.read()
            assert "What is ML?" in content
            assert "Machine Learning" in content
            assert "reference" in content
            assert "candidate" in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_questions_from_json(self):
        """Test loading questions from JSON."""
        test_questions = [
            {
                "question": "What is Python?",
                "domain": "Software Engineering",
                "difficulty": "beginner",
            },
            {
                "question": "Explain OOP",
                "domain": "Software Engineering",
                "difficulty": "intermediate",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_questions, f)
            filepath = f.name

        try:
            loaded_questions = load_questions_from_json(filepath)
            assert len(loaded_questions) == 2
            assert loaded_questions[0]["question"] == "What is Python?"
            assert loaded_questions[1]["domain"] == "Software Engineering"
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_validate_question_format(self):
        """Test question format validation."""
        # Valid question
        valid_question = {
            "question": "Test question",
            "domain": "Data Science",
            "difficulty": "intermediate",
        }
        assert validate_question_format(valid_question) == True

        # Missing required field
        invalid_question = {"domain": "Data Science", "difficulty": "intermediate"}
        assert validate_question_format(invalid_question) == False

        # Missing domain
        invalid_question2 = {"question": "Test question", "difficulty": "intermediate"}
        assert validate_question_format(invalid_question2) == False

    def test_create_evaluation_dataset(self):
        """Test evaluation dataset creation."""
        answers_list = [
            GeneratedAnswers(
                question="Test question 1",
                domain=Domain.SOFTWARE_ENGINEERING,
                difficulty=DifficultyLevel.BEGINNER,
                reference_answers=[
                    ReferenceAnswer(
                        approach="concise_definition", answer="Short answer"
                    )
                ],
                candidate_answers=[
                    CandidateAnswer(type="perfect_match", answer="Perfect answer"),
                    CandidateAnswer(type="misconception", answer="Wrong answer"),
                ],
                metadata={},
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepaths = create_evaluation_dataset(answers_list, temp_dir)

            # Check all expected files were created
            assert "complete_dataset" in filepaths
            assert "evaluation_pairs" in filepaths
            assert "evaluation_csv" in filepaths

            # Verify files exist
            for filepath in filepaths.values():
                assert os.path.exists(filepath)

            # Check content of evaluation pairs
            with open(filepaths["evaluation_pairs"], "r") as f:
                pairs_data = json.load(f)

            assert len(pairs_data) == 2  # 1 ref * 2 candidates
            assert pairs_data[0]["question"] == "Test question 1"
            assert "expected_score" in pairs_data[0]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
