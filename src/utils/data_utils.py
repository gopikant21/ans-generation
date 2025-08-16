import json
import csv
from typing import List, Dict, Any
from datetime import datetime
import os

from ..models import GeneratedAnswers, BatchGeneratedAnswers


def export_to_json(data: GeneratedAnswers, filepath: str = None) -> str:
    """
    Export generated answers to JSON file.

    Args:
        data: GeneratedAnswers object
        filepath: Optional custom filepath

    Returns:
        The filepath where data was saved
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"generated_answers_{timestamp}.json"

    # Convert to dict for JSON serialization
    data_dict = data.dict()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    return filepath


def export_batch_to_json(data: BatchGeneratedAnswers, filepath: str = None) -> str:
    """
    Export batch generated answers to JSON file.

    Args:
        data: BatchGeneratedAnswers object
        filepath: Optional custom filepath

    Returns:
        The filepath where data was saved
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"batch_generated_answers_{timestamp}.json"

    data_dict = data.dict()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    return filepath


def export_to_csv(data: GeneratedAnswers, filepath: str = None) -> str:
    """
    Export generated answers to CSV format.

    Args:
        data: GeneratedAnswers object
        filepath: Optional custom filepath

    Returns:
        The filepath where data was saved
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"generated_answers_{timestamp}.csv"

    rows = []

    # Add reference answers
    for ref_answer in data.reference_answers:
        rows.append(
            {
                "question": data.question,
                "domain": data.domain.value,
                "difficulty": data.difficulty.value,
                "answer_type": "reference",
                "answer_category": ref_answer.approach.value,
                "answer": ref_answer.answer,
            }
        )

    # Add candidate answers
    for cand_answer in data.candidate_answers:
        rows.append(
            {
                "question": data.question,
                "domain": data.domain.value,
                "difficulty": data.difficulty.value,
                "answer_type": "candidate",
                "answer_category": cand_answer.type.value,
                "answer": cand_answer.answer,
            }
        )

    # Write to CSV
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "domain",
                "difficulty",
                "answer_type",
                "answer_category",
                "answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return filepath


def load_questions_from_csv(filepath: str) -> List[Dict[str, str]]:
    """
    Load questions from CSV file.

    Expected CSV format:
    question,domain,difficulty
    "What is machine learning?","Data Science","intermediate"

    Args:
        filepath: Path to CSV file

    Returns:
        List of question dictionaries
    """
    questions = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(
                {
                    "question": row["question"],
                    "domain": row["domain"],
                    "difficulty": row.get("difficulty", "intermediate"),
                }
            )

    return questions


def load_questions_from_json(filepath: str) -> List[Dict[str, str]]:
    """
    Load questions from JSON file.

    Expected JSON format:
    [
        {
            "question": "What is machine learning?",
            "domain": "Data Science",
            "difficulty": "intermediate"
        }
    ]

    Args:
        filepath: Path to JSON file

    Returns:
        List of question dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, list) else [data]


def validate_question_format(question_data: Dict[str, str]) -> bool:
    """
    Validate that a question dictionary has required fields.

    Args:
        question_data: Dictionary with question data

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["question", "domain"]
    return all(field in question_data for field in required_fields)


def create_evaluation_dataset(
    data_list: List[GeneratedAnswers], output_dir: str = "evaluation_dataset"
) -> Dict[str, str]:
    """
    Create a comprehensive evaluation dataset from generated answers.

    Args:
        data_list: List of GeneratedAnswers objects
        output_dir: Directory to save the dataset

    Returns:
        Dictionary with filepaths of created files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for different formats
    all_data = []
    evaluation_pairs = []

    for data in data_list:
        all_data.append(data.dict())

        # Create evaluation pairs (reference vs candidate)
        for ref_answer in data.reference_answers:
            for cand_answer in data.candidate_answers:
                evaluation_pairs.append(
                    {
                        "question": data.question,
                        "domain": data.domain.value,
                        "difficulty": data.difficulty.value,
                        "reference_answer": ref_answer.answer,
                        "reference_approach": ref_answer.approach.value,
                        "candidate_answer": cand_answer.answer,
                        "candidate_type": cand_answer.type.value,
                        "expected_score": _get_expected_score(cand_answer.type.value),
                    }
                )

    # Save files
    filepaths = {}

    # Complete dataset
    complete_path = os.path.join(output_dir, "complete_dataset.json")
    with open(complete_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    filepaths["complete_dataset"] = complete_path

    # Evaluation pairs for grading systems
    pairs_path = os.path.join(output_dir, "evaluation_pairs.json")
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_pairs, f, indent=2, ensure_ascii=False)
    filepaths["evaluation_pairs"] = pairs_path

    # CSV format for analysis
    csv_path = os.path.join(output_dir, "evaluation_pairs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if evaluation_pairs:
            writer = csv.DictWriter(f, fieldnames=evaluation_pairs[0].keys())
            writer.writeheader()
            writer.writerows(evaluation_pairs)
    filepaths["evaluation_csv"] = csv_path

    return filepaths


def _get_expected_score(candidate_type: str) -> float:
    """
    Get expected score for different candidate answer types.

    Args:
        candidate_type: Type of candidate answer

    Returns:
        Expected score (0.0 to 1.0)
    """
    score_mapping = {
        "perfect_match": 1.0,
        "alternate_correct": 0.9,
        "partial_correct": 0.7,
        "misconception": 0.2,
        "off_topic": 0.1,
        "poor_quality": 0.3,
    }
    return score_mapping.get(candidate_type, 0.5)
