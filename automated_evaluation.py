import os
import json
import re
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    relevance: float = Field(..., description="Relevance score between 0 and 100")
    factual_accuracy: float = Field(..., description="Accuracy score between 0 and 100")
    completeness: float = Field(..., description="Completeness score between 0 and 100")
    overall_score: float = Field(..., description="Weighted score between 0 and 100")


class AutomatedEvaluator:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the automated evaluator

        Args:
            openai_api_key: OpenAI API key
            model_name: Model to use for evaluation
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model=model_name)

        self.prompt_template = PromptTemplate(
            input_variables=["question", "references", "student_answer"],
            template="""
You are an expert evaluator for interview responses. Compare the STUDENT ANSWER to the REFERENCE ANSWERS if available, or evaluate based on your expertise.

Question: {question}

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

Formula: overall_score = 0.35 * factual_accuracy + 0.45 * relevance + 0.2 * completeness

Respond ONLY with valid JSON in this format:
{{
  "relevance": <float>,
  "factual_accuracy": <float>,
  "completeness": <float>,
  "overall_score": <float>
}}
""",
        )

    def extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response"""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in model output: {text}")
        json_str = match.group(0)
        return json.loads(json_str)

    def evaluate_single_answer(
        self, question: str, references: List[str], student_answer: str
    ) -> dict:
        """
        Evaluate a single answer

        Args:
            question: The interview question
            references: List of reference answers
            student_answer: The candidate's answer

        Returns:
            Dictionary with evaluation scores
        """
        try:
            prompt = self.prompt_template.format(
                question=question,
                references="\n".join(references),
                student_answer=student_answer,
            )
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            result_dict = self.extract_json(response_text)
            validated = EvaluationResult(**result_dict)
            return validated.model_dump()
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return {
                "relevance": 0.0,
                "factual_accuracy": 0.0,
                "completeness": 0.0,
                "overall_score": 0.0,
                "error": str(e),
            }

    def process_response_file(self, json_file_path: str) -> Dict[str, Any]:
        """
        Process the entire response JSON file and evaluate all answers

        Args:
            json_file_path: Path to the response JSON file

        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"Loading response file: {json_file_path}")

        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        evaluation_results = {
            "metadata": {
                "evaluated_at": datetime.now().isoformat(),
                "total_questions": len(data["results"]),
                "evaluator_model": self.llm.model_name,
            },
            "question_evaluations": [],
        }

        total_questions = len(data["results"])
        logger.info(f"Starting evaluation of {total_questions} questions")

        for idx, result in enumerate(data["results"], 1):
            logger.info(
                f"Processing question {idx}/{total_questions}: {result['question'][:50]}..."
            )

            question = result["question"]
            domain = result.get("domain", "Unknown")
            difficulty = result.get("difficulty", "Unknown")

            # Extract reference answers
            reference_answers = []
            if "reference_answers" in result:
                for ref in result["reference_answers"]:
                    if isinstance(ref, dict) and "answer" in ref:
                        reference_answers.append(ref["answer"])
                    elif isinstance(ref, str):
                        reference_answers.append(ref)

            # Evaluate each candidate answer
            candidate_evaluations = []
            if "candidate_answers" in result:
                for candidate_idx, candidate in enumerate(result["candidate_answers"]):
                    candidate_answer = (
                        candidate.get("answer", "")
                        if isinstance(candidate, dict)
                        else candidate
                    )
                    candidate_type = (
                        candidate.get("type", "unknown")
                        if isinstance(candidate, dict)
                        else "unknown"
                    )

                    logger.info(f"  Evaluating candidate answer {candidate_idx + 1}")
                    evaluation = self.evaluate_single_answer(
                        question, reference_answers, candidate_answer
                    )
                    evaluation["candidate_index"] = candidate_idx
                    evaluation["candidate_type"] = candidate_type
                    candidate_evaluations.append(evaluation)

            question_evaluation = {
                "question": question,
                "domain": domain,
                "difficulty": difficulty,
                "num_reference_answers": len(reference_answers),
                "num_candidate_answers": len(candidate_evaluations),
                "candidate_evaluations": candidate_evaluations,
            }

            evaluation_results["question_evaluations"].append(question_evaluation)

        logger.info("Evaluation completed successfully")
        return evaluation_results

    def generate_summary_report(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a summary report from evaluation results

        Args:
            evaluation_results: Results from process_response_file

        Returns:
            Summary statistics and insights
        """
        all_scores = []
        domain_scores = {}
        difficulty_scores = {}
        type_scores = {}

        for question_eval in evaluation_results["question_evaluations"]:
            domain = question_eval["domain"]
            difficulty = question_eval["difficulty"]

            for candidate_eval in question_eval["candidate_evaluations"]:
                if "error" not in candidate_eval:
                    score_data = {
                        "overall_score": candidate_eval["overall_score"],
                        "relevance": candidate_eval["relevance"],
                        "factual_accuracy": candidate_eval["factual_accuracy"],
                        "completeness": candidate_eval["completeness"],
                        "domain": domain,
                        "difficulty": difficulty,
                        "type": candidate_eval.get("candidate_type", "unknown"),
                    }
                    all_scores.append(score_data)

                    # Group by domain
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(score_data)

                    # Group by difficulty
                    if difficulty not in difficulty_scores:
                        difficulty_scores[difficulty] = []
                    difficulty_scores[difficulty].append(score_data)

                    # Group by type
                    candidate_type = candidate_eval.get("candidate_type", "unknown")
                    if candidate_type not in type_scores:
                        type_scores[candidate_type] = []
                    type_scores[candidate_type].append(score_data)

        def calculate_stats(scores_list):
            if not scores_list:
                return {}

            overall_scores = [s["overall_score"] for s in scores_list]
            relevance_scores = [s["relevance"] for s in scores_list]
            accuracy_scores = [s["factual_accuracy"] for s in scores_list]
            completeness_scores = [s["completeness"] for s in scores_list]

            return {
                "count": len(scores_list),
                "overall_score": {
                    "mean": sum(overall_scores) / len(overall_scores),
                    "min": min(overall_scores),
                    "max": max(overall_scores),
                },
                "relevance": {
                    "mean": sum(relevance_scores) / len(relevance_scores),
                    "min": min(relevance_scores),
                    "max": max(relevance_scores),
                },
                "factual_accuracy": {
                    "mean": sum(accuracy_scores) / len(accuracy_scores),
                    "min": min(accuracy_scores),
                    "max": max(accuracy_scores),
                },
                "completeness": {
                    "mean": sum(completeness_scores) / len(completeness_scores),
                    "min": min(completeness_scores),
                    "max": max(completeness_scores),
                },
            }

        summary = {
            "overall_statistics": calculate_stats(all_scores),
            "by_domain": {
                domain: calculate_stats(scores)
                for domain, scores in domain_scores.items()
            },
            "by_difficulty": {
                difficulty: calculate_stats(scores)
                for difficulty, scores in difficulty_scores.items()
            },
            "by_type": {
                type_name: calculate_stats(scores)
                for type_name, scores in type_scores.items()
            },
        }

        return summary

    def save_results(self, evaluation_results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file"""
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(evaluation_results, file, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")

    def create_excel_report(self, evaluation_results: Dict[str, Any], excel_path: str):
        """Create an Excel report with detailed results"""
        # Prepare data for Excel
        rows = []
        for question_eval in evaluation_results["question_evaluations"]:
            question = question_eval["question"]
            domain = question_eval["domain"]
            difficulty = question_eval["difficulty"]

            for candidate_eval in question_eval["candidate_evaluations"]:
                if "error" not in candidate_eval:
                    row = {
                        "Question": question,
                        "Domain": domain,
                        "Difficulty": difficulty,
                        "Candidate_Index": candidate_eval["candidate_index"],
                        "Candidate_Type": candidate_eval.get(
                            "candidate_type", "unknown"
                        ),
                        "Overall_Score": candidate_eval["overall_score"],
                        "Relevance": candidate_eval["relevance"],
                        "Factual_Accuracy": candidate_eval["factual_accuracy"],
                        "Completeness": candidate_eval["completeness"],
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)

        # Create Excel file with multiple sheets
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name="Detailed_Results", index=False)

            # Summary by domain
            domain_summary = (
                df.groupby("Domain")
                .agg(
                    {
                        "Overall_Score": ["mean", "std", "count"],
                        "Relevance": "mean",
                        "Factual_Accuracy": "mean",
                        "Completeness": "mean",
                    }
                )
                .round(2)
            )
            domain_summary.to_excel(writer, sheet_name="Domain_Summary")

            # Summary by difficulty
            difficulty_summary = (
                df.groupby("Difficulty")
                .agg(
                    {
                        "Overall_Score": ["mean", "std", "count"],
                        "Relevance": "mean",
                        "Factual_Accuracy": "mean",
                        "Completeness": "mean",
                    }
                )
                .round(2)
            )
            difficulty_summary.to_excel(writer, sheet_name="Difficulty_Summary")

            # Summary by type
            type_summary = (
                df.groupby("Candidate_Type")
                .agg(
                    {
                        "Overall_Score": ["mean", "std", "count"],
                        "Relevance": "mean",
                        "Factual_Accuracy": "mean",
                        "Completeness": "mean",
                    }
                )
                .round(2)
            )
            type_summary.to_excel(writer, sheet_name="Type_Summary")

        logger.info(f"Excel report saved to: {excel_path}")


def main():
    """Main function to run the automated evaluation"""
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    INPUT_FILE = "response_1755348961176.json"
    OUTPUT_FILE = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    EXCEL_FILE = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # Initialize evaluator
    evaluator = AutomatedEvaluator(OPENAI_API_KEY)

    # Process the response file
    evaluation_results = evaluator.process_response_file(INPUT_FILE)

    # Generate summary report
    summary = evaluator.generate_summary_report(evaluation_results)
    evaluation_results["summary"] = summary

    # Save results
    evaluator.save_results(evaluation_results, OUTPUT_FILE)
    evaluator.create_excel_report(evaluation_results, EXCEL_FILE)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    overall_stats = summary["overall_statistics"]
    print(f"Total Answers Evaluated: {overall_stats['count']}")
    print(f"Average Overall Score: {overall_stats['overall_score']['mean']:.2f}")
    print(f"Average Relevance: {overall_stats['relevance']['mean']:.2f}")
    print(f"Average Factual Accuracy: {overall_stats['factual_accuracy']['mean']:.2f}")
    print(f"Average Completeness: {overall_stats['completeness']['mean']:.2f}")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Excel report saved to: {EXCEL_FILE}")


if __name__ == "__main__":
    main()
