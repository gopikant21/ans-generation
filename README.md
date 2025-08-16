# Answer Generation System

An automated system for generating multiple valid reference answers and candidate answers for evaluation purposes using LangChain and FastAPI.

## Features

- Generate 3-5 reference answers per question with different approaches
- Generate 6+ candidate answers with varying quality levels
- Support for multiple domains (Software Engineering, Data Science, HR, etc.)
- RESTful API with FastAPI
- Configurable LLM provider (Google Gemini)
- Automated evaluation dataset creation

## Setup

### Prerequisites

- Python 3.8+
- Google API key for Gemini

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ans-generation
```

2. Create a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create environment file:

```bash
cp .env.example .env
```

5. Configure your API keys in `.env`:

```
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

## Usage

### Running the Server

Start the FastAPI server:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

### Example API Usage

#### Generate answers for a single question:

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "domain": "Data Science",
    "difficulty": "intermediate"
  }'
```

#### Batch generate answers:

```bash
curl -X POST "http://localhost:8000/api/v1/generate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      {
        "question": "What is machine learning?",
        "domain": "Data Science",
        "difficulty": "intermediate"
      },
      {
        "question": "Explain object-oriented programming",
        "domain": "Software Engineering",
        "difficulty": "beginner"
      }
    ]
  }'
```

### Python SDK Usage

```python
from src.answer_generator import AnswerGenerator

# Initialize generator
generator = AnswerGenerator(model_name="gemini-2.5-pro")

# Generate answers
result = generator.generate_answers(
    question="What is machine learning?",
    domain="Data Science",
    difficulty="intermediate"
)

print(result)
```

## Configuration

### Supported Domains

- Software Engineering
- Data Science
- Machine Learning
- Human Resources
- General Knowledge
- Finance
- Healthcare
- Education

### Difficulty Levels

- `beginner`: Basic concepts and definitions
- `intermediate`: Applied knowledge and reasoning
- `advanced`: Complex analysis and expert-level understanding

### Answer Types

#### Reference Answers:

- `concise_definition`: Brief, accurate definitions
- `detailed_explanation`: Comprehensive explanations
- `step_by_step_solution`: Procedural approaches
- `analogy_or_example`: Example-driven explanations
- `formula_based`: Mathematical or technical formulations

#### Candidate Answers:

- `perfect_match`: Nearly identical to reference
- `alternate_correct`: Valid but different approach
- `partial_correct`: Some correct elements, missing details
- `misconception`: Common but incorrect understanding
- `off_topic`: Plausible but irrelevant
- `poor_quality`: Poorly structured or vague

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Project Structure

```
ans-generation/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── answer_generator.py     # Core generation logic
│   ├── models.py              # Pydantic models
│   ├── prompts/               # LLM prompts
│   └── utils/                 # Utility functions
├── tests/                     # Test files
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## API Endpoints

### POST /api/v1/generate

Generate answers for a single question.

**Request Body:**

```json
{
  "question": "string",
  "domain": "string",
  "difficulty": "beginner|intermediate|advanced"
}
```

### POST /api/v1/generate/batch

Generate answers for multiple questions.

**Request Body:**

```json
{
  "questions": [
    {
      "question": "string",
      "domain": "string",
      "difficulty": "string"
    }
  ]
}
```

### GET /api/v1/domains

Get list of supported domains.

### GET /api/v1/health

Health check endpoint.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and ensure they pass
6. Submit a pull request

## License

MIT License

## Support

For issues and questions, please open a GitHub issue.
