import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.main import app
from src.models import GeneratedAnswers, Domain, DifficultyLevel


class TestAPI:
    """Test cases for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_generated_answers(self):
        """Mock GeneratedAnswers for testing."""
        from src.models import ReferenceAnswer, CandidateAnswer

        return GeneratedAnswers(
            question="What is machine learning?",
            domain=Domain.DATA_SCIENCE,
            difficulty=DifficultyLevel.INTERMEDIATE,
            reference_answers=[
                ReferenceAnswer(
                    approach="concise_definition", answer="ML is AI subset"
                ),
                ReferenceAnswer(
                    approach="detailed_explanation", answer="Detailed explanation..."
                ),
            ],
            candidate_answers=[
                CandidateAnswer(type="perfect_match", answer="Perfect match answer"),
                CandidateAnswer(type="misconception", answer="Wrong answer"),
            ],
            metadata={"model_used": "gpt-4"},
        )

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_domains_endpoint(self, client):
        """Test domains endpoint."""
        response = client.get("/api/v1/domains")
        assert response.status_code == 200
        domains = response.json()
        assert isinstance(domains, list)
        assert "Software Engineering" in domains
        assert "Data Science" in domains

    def test_generate_endpoint_success(self, client, mock_generated_answers):
        """Test successful answer generation."""
        with patch(
            "src.main.answer_generator.generate_answers",
            return_value=mock_generated_answers,
        ):
            response = client.post(
                "/api/v1/generate",
                json={
                    "question": "What is machine learning?",
                    "domain": "Data Science",
                    "difficulty": "intermediate",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["question"] == "What is machine learning?"
            assert data["domain"] == "Data Science"
            assert "reference_answers" in data
            assert "candidate_answers" in data

    def test_generate_endpoint_invalid_domain(self, client):
        """Test generation with invalid domain."""
        response = client.post(
            "/api/v1/generate",
            json={
                "question": "Test question",
                "domain": "Invalid Domain",
                "difficulty": "intermediate",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_generate_endpoint_missing_question(self, client):
        """Test generation with missing question."""
        response = client.post(
            "/api/v1/generate",
            json={"domain": "Data Science", "difficulty": "intermediate"},
        )

        assert response.status_code == 422  # Validation error

    def test_batch_generate_endpoint(self, client, mock_generated_answers):
        """Test batch generation endpoint."""
        from src.models import BatchGeneratedAnswers

        mock_batch_result = BatchGeneratedAnswers(
            results=[mock_generated_answers],
            total_questions=1,
            successful_generations=1,
            failed_generations=0,
        )

        with patch(
            "src.main.batch_generator.generate_batch",
            return_value=[mock_generated_answers],
        ):
            response = client.post(
                "/api/v1/generate/batch",
                json={
                    "questions": [
                        {
                            "question": "What is machine learning?",
                            "domain": "Data Science",
                            "difficulty": "intermediate",
                        }
                    ]
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total_questions"] == 1
            assert data["successful_generations"] == 1
            assert len(data["results"]) == 1

    def test_batch_generate_empty_questions(self, client):
        """Test batch generation with empty questions list."""
        response = client.post("/api/v1/generate/batch", json={"questions": []})

        assert response.status_code == 400
        data = response.json()
        assert "No questions provided" in data["message"]

    def test_batch_generate_too_many_questions(self, client):
        """Test batch generation with too many questions."""
        questions = [
            {
                "question": f"Question {i}",
                "domain": "Data Science",
                "difficulty": "intermediate",
            }
            for i in range(51)  # Exceeds limit of 50
        ]

        response = client.post("/api/v1/generate/batch", json={"questions": questions})

        assert response.status_code == 400
        data = response.json()
        assert "cannot exceed 50" in data["message"]

    def test_generate_endpoint_server_error(self, client):
        """Test server error handling."""
        with patch(
            "src.main.answer_generator.generate_answers",
            side_effect=Exception("Test error"),
        ):
            response = client.post(
                "/api/v1/generate",
                json={
                    "question": "Test question",
                    "domain": "Data Science",
                    "difficulty": "intermediate",
                },
            )

            assert response.status_code == 500
            data = response.json()
            assert data["error"] == "Internal Server Error"

    def test_async_generate_endpoint(self, client):
        """Test async generation endpoint."""
        response = client.post(
            "/api/v1/generate/async",
            json={
                "question": "What is machine learning?",
                "domain": "Data Science",
                "difficulty": "intermediate",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "started"


if __name__ == "__main__":
    pytest.main([__file__])
