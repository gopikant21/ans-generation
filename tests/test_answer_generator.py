import pytest
import os
from unittest.mock import Mock, patch
from src.answer_generator import AnswerGenerator
from src.models import Domain, DifficultyLevel


class TestAnswerGenerator:
    """Test cases for AnswerGenerator class."""

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing."""
        mock_response = Mock()
        mock_response.content = """[
            {"approach": "concise_definition", "answer": "Machine learning is a subset of AI that enables computers to learn from data."},
            {"approach": "detailed_explanation", "answer": "Machine learning is a field of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed for every scenario."},
            {"approach": "step_by_step_solution", "answer": "1. Data collection 2. Data preprocessing 3. Model selection 4. Training 5. Evaluation 6. Deployment"},
            {"approach": "analogy_or_example", "answer": "Machine learning is like teaching a child to recognize animals by showing them many pictures, rather than describing every detail of each animal."}
        ]"""
        return mock_response

    @pytest.fixture
    def generator(self):
        """Create AnswerGenerator instance for testing."""
        with patch("src.answer_generator.genai.GenerativeModel"):
            return AnswerGenerator(model_name="gemini-2.5-pro-test")

    def test_initialization(self):
        """Test AnswerGenerator initialization."""
        with patch("src.answer_generator.genai.GenerativeModel") as mock_genai:
            generator = AnswerGenerator(model_name="gemini-2.5-pro", temperature=0.5)
            assert generator.model_name == "gemini-2.5-pro"
            assert generator.temperature == 0.5
            assert generator.max_retries == 3
            mock_genai.assert_called_once()

    def test_generate_reference_answers(self, generator, mock_llm_response):
        """Test reference answer generation."""
        with patch.object(
            generator, "_call_llm_with_retry", return_value=mock_llm_response
        ):
            answers = generator.generate_reference_answers(
                "What is machine learning?",
                Domain.DATA_SCIENCE,
                DifficultyLevel.INTERMEDIATE,
            )

            assert len(answers) == 4
            assert answers[0].approach == "concise_definition"
            assert "machine learning" in answers[0].answer.lower()

    def test_generate_candidate_answers(self, generator):
        """Test candidate answer generation."""
        mock_response = Mock()
        mock_response.content = """[
            {"type": "perfect_match", "answer": "Machine learning is a subset of AI."},
            {"type": "alternate_correct", "answer": "ML enables computers to learn patterns from data."},
            {"type": "partial_correct", "answer": "Machine learning uses algorithms to analyze data."},
            {"type": "misconception", "answer": "Machine learning is the same as artificial intelligence."},
            {"type": "off_topic", "answer": "Data science involves statistical analysis."},
            {"type": "poor_quality", "answer": "ML is like... you know... computer stuff."}
        ]"""

        with patch.object(
            generator, "_call_llm_with_retry", return_value=mock_response
        ):
            from src.models import ReferenceAnswer

            ref_answers = [
                ReferenceAnswer(approach="concise_definition", answer="Test reference")
            ]

            answers = generator.generate_candidate_answers(
                "What is machine learning?",
                Domain.DATA_SCIENCE,
                DifficultyLevel.INTERMEDIATE,
                ref_answers,
            )

            assert len(answers) == 6
            assert answers[0].type == "perfect_match"
            assert answers[3].type == "misconception"

    def test_generate_answers_complete(self, generator, mock_llm_response):
        """Test complete answer generation."""
        # Mock both reference and candidate answer generation
        candidate_mock = Mock()
        candidate_mock.content = """[
            {"type": "perfect_match", "answer": "ML is AI subset"},
            {"type": "alternate_correct", "answer": "Computers learn from data"},
            {"type": "partial_correct", "answer": "Uses algorithms"},
            {"type": "misconception", "answer": "ML equals AI"},
            {"type": "off_topic", "answer": "Statistics is important"},
            {"type": "poor_quality", "answer": "Computer thing"}
        ]"""

        with patch.object(
            generator,
            "_call_llm_with_retry",
            side_effect=[mock_llm_response, candidate_mock],
        ):
            result = generator.generate_answers(
                "What is machine learning?", "Data Science", "intermediate"
            )

            assert result.question == "What is machine learning?"
            assert result.domain == Domain.DATA_SCIENCE
            assert result.difficulty == DifficultyLevel.INTERMEDIATE
            assert len(result.reference_answers) == 4
            assert len(result.candidate_answers) == 6
            assert "model_used" in result.metadata

    def test_llm_retry_logic(self, generator):
        """Test LLM retry logic on failures."""
        mock_messages = [Mock()]

        # Mock LLM to fail twice then succeed
        with patch.object(generator, "llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                Exception("API Error"),
                Exception("Rate Limit"),
                Mock(),  # Success on third try
            ]

            result = generator._call_llm_with_retry(mock_messages)
            assert mock_llm.invoke.call_count == 3
            assert result is not None

    def test_llm_retry_exhaustion(self, generator):
        """Test LLM retry logic when all attempts fail."""
        mock_messages = [Mock()]

        with patch.object(generator, "llm") as mock_llm:
            mock_llm.invoke.side_effect = Exception("Persistent Error")

            with pytest.raises(Exception, match="Persistent Error"):
                generator._call_llm_with_retry(mock_messages)

            assert mock_llm.invoke.call_count == 3  # max_retries

    def test_invalid_json_handling(self, generator):
        """Test handling of invalid JSON responses."""
        invalid_response = Mock()
        invalid_response.content = "This is not valid JSON"

        with patch.object(
            generator, "_call_llm_with_retry", return_value=invalid_response
        ):
            with pytest.raises(
                ValueError, match="Failed to parse LLM response as JSON"
            ):
                generator.generate_reference_answers(
                    "Test question",
                    Domain.SOFTWARE_ENGINEERING,
                    DifficultyLevel.BEGINNER,
                )


if __name__ == "__main__":
    pytest.main([__file__])
