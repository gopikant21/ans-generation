import os
import json
import logging
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

from .models import (
    QuestionInput,
    GeneratedAnswers,
    ReferenceAnswer,
    CandidateAnswer,
    Domain,
    DifficultyLevel,
    ReferenceApproach,
    CandidateType,
)
from .prompts.templates import (
    REFERENCE_ANSWERS_PROMPT,
    CANDIDATE_ANSWERS_PROMPT,
    get_additional_approach,
    get_domain_context,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Core class for generating reference and candidate answers using Google Gemini.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._initialize_genai()

    def _initialize_genai(self):
        """Initialize Google Generative AI."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            self.client = genai.Client(api_key=api_key)
            logger.info(f"Initialized Gemini client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt to reduce likelihood of safety filter blocks."""
        # Add educational context to make intent clear
        educational_prefix = """This is an educational content generation request for academic purposes. 
Please provide accurate, helpful information that would be appropriate for educational use.

"""
        return educational_prefix + prompt

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON content from markdown-formatted response."""
        # Remove leading/trailing whitespace
        response_text = response_text.strip()

        # Check if response is wrapped in markdown code blocks
        if response_text.startswith("```json"):
            # Find the start and end of the JSON content
            start_idx = response_text.find("```json") + len("```json")
            end_idx = response_text.rfind("```")
            if end_idx > start_idx:
                response_text = response_text[start_idx:end_idx].strip()
        elif response_text.startswith("```"):
            # Handle case where it's just ``` without json
            start_idx = response_text.find("```") + 3
            end_idx = response_text.rfind("```")
            if end_idx > start_idx:
                response_text = response_text[start_idx:end_idx].strip()

        return response_text

    def _call_gemini_with_retry(self, prompt: str, response_schema=None) -> Any:
        """Call Gemini API with retry logic and optional structured output."""
        # Sanitize the prompt for better safety filter compatibility
        sanitized_prompt = self._sanitize_prompt(prompt)

        for attempt in range(self.max_retries):
            try:
                config = {
                    "temperature": self.temperature,
                    "max_output_tokens": 4000,
                }

                # Add structured output configuration if schema provided
                if response_schema:
                    config["response_mime_type"] = "application/json"
                    config["response_schema"] = response_schema

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=sanitized_prompt,
                    config=config,
                )

                # Check if response is valid
                if not response or not hasattr(response, "text"):
                    logger.error("Empty or invalid response from Gemini")
                    raise ValueError("Empty response from Gemini")

                # If structured output requested, try parsed first, fallback to manual parsing
                if response_schema:
                    if hasattr(response, "parsed") and response.parsed is not None:
                        logger.info(f"Using structured output: {response.parsed}")
                        return response.parsed
                    else:
                        # Fallback to manual JSON parsing
                        logger.warning(
                            "Structured output parsing failed, falling back to manual parsing"
                        )
                        try:
                            json_text = self._extract_json_from_response(response.text)
                            parsed_data = json.loads(json_text)

                            # Convert to appropriate Pydantic models
                            if response_schema == list[ReferenceAnswer]:
                                return [ReferenceAnswer(**item) for item in parsed_data]
                            elif response_schema == list[CandidateAnswer]:
                                return [CandidateAnswer(**item) for item in parsed_data]
                            else:
                                return parsed_data
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.error(f"Failed to parse JSON manually: {e}")
                            logger.error(f"Raw response: {response.text}")
                            raise ValueError(
                                f"Failed to parse structured response: {e}"
                            )
                else:
                    return response.text

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                # Wait before retry (exponential backoff)
                time.sleep(2**attempt)

        # Return appropriate default values
        if response_schema:
            return []
        return ""

    def generate_reference_answers(
        self, question: str, domain: Domain, difficulty: DifficultyLevel
    ) -> List[ReferenceAnswer]:
        """Generate multiple reference answers for a question."""
        try:
            # Get domain-specific context
            domain_context = get_domain_context(domain.value)
            additional_approach, additional_json = get_additional_approach(
                domain.value, difficulty.value
            )

            num_references = 5 if additional_approach else 4

            # Format the prompt with system context
            system_context = (
                f"You are an expert in {domain.value}. {domain_context['context']}"
            )
            user_prompt = REFERENCE_ANSWERS_PROMPT.format(
                question=question,
                domain=domain.value,
                difficulty=difficulty.value,
                num_references=num_references,
                additional_approach=additional_approach,
                additional_approach_json=additional_json,
            )

            full_prompt = f"{system_context}\n\n{user_prompt}"

            # Generate answers with structured output
            answers_data = self._call_gemini_with_retry(
                full_prompt, response_schema=list[ReferenceAnswer]
            )

            # Validate response
            if not answers_data or not isinstance(answers_data, list):
                logger.error(f"Invalid response format: {answers_data}")
                raise ValueError("Failed to generate valid reference answers")

            logger.info(f"Generated {len(answers_data)} reference answers")
            return answers_data

        except Exception as e:
            logger.error(f"Error generating reference answers: {e}")
            raise

    def generate_candidate_answers(
        self,
        question: str,
        domain: Domain,
        difficulty: DifficultyLevel,
        reference_answers: List[ReferenceAnswer],
    ) -> List[CandidateAnswer]:
        """Generate diverse candidate answers for evaluation testing."""
        try:
            # Validate input
            if not reference_answers:
                raise ValueError("No reference answers provided")

            # Format reference answers for context
            ref_answers_text = "\n".join(
                [f"- {ref.approach.value}: {ref.answer}" for ref in reference_answers]
            )

            domain_context = get_domain_context(domain.value)

            # Format the prompt with system context
            system_context = f"You are creating test cases for {domain.value} evaluation. {domain_context['context']} Common misconceptions include: {domain_context['common_misconceptions']}"
            user_prompt = CANDIDATE_ANSWERS_PROMPT.format(
                question=question,
                domain=domain.value,
                difficulty=difficulty.value,
                reference_answers=ref_answers_text,
            )

            full_prompt = f"{system_context}\n\n{user_prompt}"

            # Generate answers with structured output
            answers_data = self._call_gemini_with_retry(
                full_prompt, response_schema=list[CandidateAnswer]
            )

            # Validate response
            if not answers_data or not isinstance(answers_data, list):
                logger.error(f"Invalid response format: {answers_data}")
                raise ValueError("Failed to generate valid candidate answers")

            logger.info(f"Generated {len(answers_data)} candidate answers")
            return answers_data

        except Exception as e:
            logger.error(f"Error generating candidate answers: {e}")
            raise

    def generate_answers(
        self, question: str, domain: str, difficulty: str = "intermediate"
    ) -> GeneratedAnswers:
        """Generate complete answer set for a question."""
        try:
            # Convert string inputs to enums
            domain_enum = Domain(domain)
            difficulty_enum = DifficultyLevel(difficulty)

            logger.info(f"Generating answers for question in {domain} domain")

            # Generate reference answers
            logger.info("Generating reference answers...")
            reference_answers = self.generate_reference_answers(
                question, domain_enum, difficulty_enum
            )

            # Generate candidate answers
            logger.info("Generating candidate answers...")
            candidate_answers = self.generate_candidate_answers(
                question, domain_enum, difficulty_enum, reference_answers
            )

            # Create metadata
            metadata = {
                "model_used": self.model_name,
                "temperature": self.temperature,
                "num_reference_answers": len(reference_answers),
                "num_candidate_answers": len(candidate_answers),
            }

            return GeneratedAnswers(
                question=question,
                domain=domain_enum,
                difficulty=difficulty_enum,
                reference_answers=reference_answers,
                candidate_answers=candidate_answers,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error in generate_answers: {e}")
            raise

    async def generate_answers_async(
        self, question: str, domain: str, difficulty: str = "intermediate"
    ) -> GeneratedAnswers:
        """Async version of generate_answers."""
        # For now, we'll use sync implementation
        # In production, you'd want to use async LLM calls
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_answers, question, domain, difficulty
        )


class BatchAnswerGenerator:
    """
    Class for handling batch generation of answers.
    """

    def __init__(self, generator: AnswerGenerator):
        self.generator = generator

    def generate_batch(
        self, questions: List[QuestionInput], max_workers: int = 3
    ) -> List[Optional[GeneratedAnswers]]:
        """Generate answers for multiple questions in parallel."""
        import concurrent.futures

        results = []
        successful = 0
        failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(
                    self.generator.generate_answers,
                    q.question,
                    q.domain.value,
                    q.difficulty.value,
                ): q
                for q in questions
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                    successful += 1
                    logger.info(
                        f"Successfully generated answers for: {question.question[:50]}..."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate answers for question: {question.question[:50]}... Error: {e}"
                    )
                    results.append(None)
                    failed += 1

        logger.info(
            f"Batch generation complete. Successful: {successful}, Failed: {failed}"
        )
        return results
