import os
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from .models import (
    QuestionInput,
    BatchQuestionInput,
    GeneratedAnswers,
    BatchGeneratedAnswers,
    HealthResponse,
    ErrorResponse,
    Domain,
)
from .answer_generator import AnswerGenerator, BatchAnswerGenerator

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Answer Generation System",
    description="Automated system for generating reference and candidate answers for evaluation purposes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generators
answer_generator = AnswerGenerator(
    model_name=os.getenv("DEFAULT_MODEL", "gpt-4"), temperature=0.7
)
batch_generator = BatchAnswerGenerator(answer_generator)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Answer Generation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", version="1.0.0", timestamp=datetime.utcnow().isoformat()
    )


@app.get("/api/v1/domains", response_model=List[str])
async def get_domains():
    """Get list of supported domains."""
    return [domain.value for domain in Domain]


@app.post("/api/v1/generate", response_model=GeneratedAnswers)
async def generate_answers(question_input: QuestionInput):
    """
    Generate reference and candidate answers for a single question.

    - **question**: The question to generate answers for
    - **domain**: The subject domain (e.g., Software Engineering, Data Science)
    - **difficulty**: Difficulty level (beginner, intermediate, advanced)
    """
    try:
        result = answer_generator.generate_answers(
            question=question_input.question,
            domain=question_input.domain.value,
            difficulty=question_input.difficulty.value,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/generate/batch", response_model=BatchGeneratedAnswers)
async def generate_batch_answers(batch_input: BatchQuestionInput):
    """
    Generate answers for multiple questions in batch.

    - **questions**: List of questions with their domains and difficulty levels
    """
    try:
        if not batch_input.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        if len(batch_input.questions) > 50:  # Limit batch size
            raise HTTPException(
                status_code=400, detail="Batch size cannot exceed 50 questions"
            )

        results = batch_generator.generate_batch(batch_input.questions)

        # Filter out None results and count successes/failures
        successful_results = [r for r in results if r is not None]
        successful_count = len(successful_results)
        failed_count = len(results) - successful_count

        return BatchGeneratedAnswers(
            results=successful_results,
            total_questions=len(batch_input.questions),
            successful_generations=successful_count,
            failed_generations=failed_count,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/generate/async")
async def generate_answers_async(
    question_input: QuestionInput, background_tasks: BackgroundTasks
):
    """
    Start asynchronous answer generation (for long-running requests).
    Returns a task ID that can be used to check status.
    """
    # This would typically integrate with a task queue like Celery
    # For now, we'll return a simple response
    task_id = f"task_{datetime.utcnow().timestamp()}"

    # Add background task
    background_tasks.add_task(_background_generate, question_input, task_id)

    return {
        "task_id": task_id,
        "status": "started",
        "message": "Answer generation started in background",
    }


async def _background_generate(question_input: QuestionInput, task_id: str):
    """Background task for answer generation."""
    try:
        # In production, you'd store this in a database or cache
        result = await answer_generator.generate_answers_async(
            question=question_input.question,
            domain=question_input.domain.value,
            difficulty=question_input.difficulty.value,
        )
        # Store result with task_id for later retrieval
        print(f"Background task {task_id} completed successfully")
    except Exception as e:
        print(f"Background task {task_id} failed: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}", message=str(exc.detail)
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            details={"exception": str(exc)} if os.getenv("DEBUG") == "true" else None,
        ).dict(),
    )


if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
    )
