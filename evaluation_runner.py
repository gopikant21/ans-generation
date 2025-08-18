#!/usr/bin/env python3
"""
Comprehensive Interview Answer Evaluation System

This script automates the evaluation of interview responses using OpenAI's language models.
It processes JSON files containing questions, reference answers, and candidate responses,
then generates detailed evaluation reports.

Usage:
    python evaluation_runner.py [--config config.py] [--input input.json] [--output output_dir]
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import the evaluation components
from automated_evaluation import AutomatedEvaluator
from evaluation_config import EVALUATION_CONFIG, EVALUATION_PROMPTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Enhanced evaluator with progress tracking and robust error handling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.evaluator = AutomatedEvaluator(
            self.openai_api_key, model_name=config.get("openai_model", "gpt-4o-mini")
        )

        # Create output directory
        self.output_dir = Path(config.get("output_directory", "evaluation_results"))
        self.output_dir.mkdir(exist_ok=True)

        # Progress tracking
        self.progress_file = self.output_dir / "evaluation_progress.json"
        self.completed_evaluations = self.load_progress()

    def load_progress(self) -> Dict[str, Any]:
        """Load existing progress if available"""
        if self.progress_file.exists() and self.config.get("save_progress", True):
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return {"completed": [], "failed": [], "metadata": {}}

    def save_progress(self, evaluation_results: List[Dict]):
        """Save current progress"""
        if self.config.get("save_progress", True):
            progress_data = {
                "completed": evaluation_results,
                "failed": getattr(self, "failed_evaluations", []),
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_completed": len(evaluation_results),
                },
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

    def evaluate_with_enhanced_prompt(
        self,
        question: str,
        domain: str,
        difficulty: str,
        references: List[str],
        student_answer: str,
    ) -> Dict[str, Any]:
        """Evaluate using enhanced prompt template"""
        weights = self.config.get("scoring_weights", {})

        prompt = EVALUATION_PROMPTS["main_template"].format(
            question=question,
            domain=domain,
            difficulty=difficulty,
            references="\n".join(references),
            student_answer=student_answer,
            factual_weight=weights.get("factual_accuracy", 0.35),
            relevance_weight=weights.get("relevance", 0.45),
            completeness_weight=weights.get("completeness", 0.20),
        )

        try:
            response = self.evaluator.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            result_dict = self.evaluator.extract_json(response_text)

            # Add grade based on overall score
            overall_score = result_dict.get("overall_score", 0)
            thresholds = self.config.get("score_thresholds", {})

            if overall_score >= thresholds.get("excellent", 90):
                grade = "A"
            elif overall_score >= thresholds.get("good", 80):
                grade = "B"
            elif overall_score >= thresholds.get("satisfactory", 70):
                grade = "C"
            elif overall_score >= thresholds.get("below_average", 60):
                grade = "D"
            else:
                grade = "F"

            result_dict["grade"] = grade
            result_dict["evaluation_timestamp"] = datetime.now().isoformat()

            return result_dict

        except Exception as e:
            logger.error(f"Error in enhanced evaluation: {e}")
            return {
                "relevance": 0.0,
                "factual_accuracy": 0.0,
                "completeness": 0.0,
                "overall_score": 0.0,
                "grade": "F",
                "error": str(e),
                "strengths": "Could not evaluate",
                "weaknesses": "Evaluation failed",
            }

    def process_json_file(self, json_file_path: str) -> List[Dict[str, Any]]:
        """Process JSON file with enhanced error handling and progress tracking"""
        logger.info(f"Processing file: {json_file_path}")

        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        all_evaluations = []
        failed_evaluations = []
        total_questions = len(data["results"])

        # Create progress bar
        with tqdm(total=total_questions, desc="Evaluating questions") as pbar:
            for idx, result in enumerate(data["results"]):
                question = result["question"]
                domain = result.get("domain", "Unknown")
                difficulty = result.get("difficulty", "Unknown")

                pbar.set_description(f"Q{idx+1}: {question[:30]}...")

                # Extract reference answers
                reference_answers = []
                if "reference_answers" in result and result["reference_answers"]:
                    for ref in result["reference_answers"]:
                        if isinstance(ref, dict) and "answer" in ref:
                            reference_answers.append(ref["answer"])
                        elif isinstance(ref, str):
                            reference_answers.append(ref)

                if not reference_answers:
                    logger.warning(f"No reference answers for question {idx+1}")
                    pbar.update(1)
                    continue

                # Process candidate answers
                if "candidate_answers" in result and result["candidate_answers"]:
                    question_evaluations = []

                    for candidate_idx, candidate in enumerate(
                        result["candidate_answers"]
                    ):
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

                        try:
                            evaluation = self.evaluate_with_enhanced_prompt(
                                question,
                                domain,
                                difficulty,
                                reference_answers,
                                candidate_answer,
                            )

                            # Add metadata
                            evaluation.update(
                                {
                                    "question_index": idx,
                                    "question": question,
                                    "domain": domain,
                                    "difficulty": difficulty,
                                    "candidate_index": candidate_idx,
                                    "candidate_type": candidate_type,
                                    "candidate_answer_preview": (
                                        candidate_answer[:200] + "..."
                                        if len(candidate_answer) > 200
                                        else candidate_answer
                                    ),
                                    "num_reference_answers": len(reference_answers),
                                }
                            )

                            question_evaluations.append(evaluation)

                        except Exception as e:
                            error_info = {
                                "question_index": idx,
                                "candidate_index": candidate_idx,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                            failed_evaluations.append(error_info)
                            logger.error(
                                f"Failed to evaluate Q{idx+1}, C{candidate_idx+1}: {e}"
                            )

                    all_evaluations.extend(question_evaluations)

                    # Save progress periodically
                    if len(all_evaluations) % self.config.get("batch_size", 5) == 0:
                        self.save_progress(all_evaluations)

                pbar.update(1)

        self.failed_evaluations = failed_evaluations
        logger.info(
            f"Completed {len(all_evaluations)} evaluations, {len(failed_evaluations)} failed"
        )

        return all_evaluations

    def generate_comprehensive_report(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        if not evaluations:
            return {}

        df = pd.DataFrame(evaluations)

        # Basic statistics
        numeric_cols = [
            "overall_score",
            "relevance",
            "factual_accuracy",
            "completeness",
        ]
        basic_stats = df[numeric_cols].describe()

        # Group analyses
        domain_analysis = df.groupby("domain")[numeric_cols].agg(
            ["mean", "std", "count"]
        )
        difficulty_analysis = df.groupby("difficulty")[numeric_cols].agg(
            ["mean", "std", "count"]
        )
        type_analysis = df.groupby("candidate_type")[numeric_cols].agg(
            ["mean", "std", "count"]
        )
        grade_distribution = df["grade"].value_counts()

        # Performance insights
        top_performers = df.nlargest(10, "overall_score")[
            ["question_index", "candidate_type", "overall_score", "grade"]
        ]
        bottom_performers = df.nsmallest(10, "overall_score")[
            ["question_index", "candidate_type", "overall_score", "grade"]
        ]

        # Correlation analysis
        correlation_matrix = df[numeric_cols].corr()

        report = {
            "basic_statistics": basic_stats.to_dict(),
            "domain_analysis": domain_analysis.to_dict(),
            "difficulty_analysis": difficulty_analysis.to_dict(),
            "type_analysis": type_analysis.to_dict(),
            "grade_distribution": grade_distribution.to_dict(),
            "top_performers": top_performers.to_dict("records"),
            "bottom_performers": bottom_performers.to_dict("records"),
            "correlations": correlation_matrix.to_dict(),
            "total_evaluations": len(evaluations),
            "unique_questions": df["question_index"].nunique(),
            "unique_domains": df["domain"].nunique(),
        }

        return report

    def create_visualizations(self, evaluations: List[Dict]):
        """Create comprehensive visualizations"""
        if not evaluations or not self.config.get("generate_plots", True):
            return

        df = pd.DataFrame(evaluations)

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Interview Evaluation Analysis Dashboard", fontsize=16, fontweight="bold"
        )

        # 1. Score distribution
        axes[0, 0].hist(
            df["overall_score"], bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Overall Score Distribution")
        axes[0, 0].set_xlabel("Overall Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            df["overall_score"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {df["overall_score"].mean():.1f}',
        )
        axes[0, 0].legend()

        # 2. Performance by domain
        domain_scores = (
            df.groupby("domain")["overall_score"].mean().sort_values(ascending=True)
        )
        domain_scores.plot(kind="barh", ax=axes[0, 1], color="lightcoral")
        axes[0, 1].set_title("Average Performance by Domain")
        axes[0, 1].set_xlabel("Average Overall Score")

        # 3. Performance by difficulty
        difficulty_scores = df.groupby("difficulty")["overall_score"].mean()
        difficulty_scores.plot(kind="bar", ax=axes[0, 2], color="lightgreen")
        axes[0, 2].set_title("Average Performance by Difficulty")
        axes[0, 2].set_ylabel("Average Overall Score")
        axes[0, 2].tick_params(axis="x", rotation=45)

        # 4. Grade distribution
        grade_counts = df["grade"].value_counts().sort_index()
        axes[1, 0].pie(
            grade_counts.values,
            labels=grade_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1, 0].set_title("Grade Distribution")

        # 5. Score components comparison
        score_components = df[["relevance", "factual_accuracy", "completeness"]].mean()
        score_components.plot(
            kind="bar", ax=axes[1, 1], color=["orange", "purple", "brown"]
        )
        axes[1, 1].set_title("Average Score Components")
        axes[1, 1].set_ylabel("Average Score")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. Performance by candidate type
        if "candidate_type" in df.columns:
            type_scores = (
                df.groupby("candidate_type")["overall_score"]
                .mean()
                .sort_values(ascending=True)
            )
            if len(type_scores) > 0:
                type_scores.plot(kind="barh", ax=axes[1, 2], color="gold")
                axes[1, 2].set_title("Average Performance by Candidate Type")
                axes[1, 2].set_xlabel("Average Overall Score")

        plt.tight_layout()

        # Save the plot
        plot_file = (
            self.output_dir
            / f'evaluation_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to: {plot_file}")

        plt.show()

    def run_complete_evaluation(self, input_file: str) -> str:
        """Run the complete evaluation pipeline"""
        logger.info("Starting comprehensive evaluation pipeline...")

        # Process evaluations
        evaluations = self.process_json_file(input_file)

        if not evaluations:
            logger.error("No evaluations completed successfully")
            return None

        # Generate comprehensive report
        report = self.generate_comprehensive_report(evaluations)

        # Create visualizations
        self.create_visualizations(evaluations)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Detailed results
        detailed_file = self.output_dir / f"detailed_evaluation_{timestamp}.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "evaluations": evaluations,
                    "comprehensive_report": report,
                    "configuration": self.config,
                    "metadata": {
                        "evaluated_at": datetime.now().isoformat(),
                        "total_evaluations": len(evaluations),
                        "failed_evaluations": len(
                            getattr(self, "failed_evaluations", [])
                        ),
                        "input_file": input_file,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Excel report
        excel_file = self.output_dir / f"evaluation_report_{timestamp}.xlsx"
        self.create_excel_report(evaluations, report, excel_file)

        # Summary report
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.txt"
        self.create_text_summary(report, summary_file)

        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")

        return str(detailed_file)

    def create_excel_report(
        self, evaluations: List[Dict], report: Dict, excel_file: Path
    ):
        """Create comprehensive Excel report"""
        df = pd.DataFrame(evaluations)

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Main results
            df.to_excel(writer, sheet_name="Detailed_Results", index=False)

            # Summary statistics
            summary_df = pd.DataFrame(report["basic_statistics"])
            summary_df.to_excel(writer, sheet_name="Summary_Statistics")

            # Domain analysis
            domain_df = pd.DataFrame(report["domain_analysis"])
            domain_df.to_excel(writer, sheet_name="Domain_Analysis")

            # Difficulty analysis
            difficulty_df = pd.DataFrame(report["difficulty_analysis"])
            difficulty_df.to_excel(writer, sheet_name="Difficulty_Analysis")

            # Grade distribution
            grade_df = pd.DataFrame(
                list(report["grade_distribution"].items()), columns=["Grade", "Count"]
            )
            grade_df.to_excel(writer, sheet_name="Grade_Distribution", index=False)

            # Top and bottom performers
            top_df = pd.DataFrame(report["top_performers"])
            top_df.to_excel(writer, sheet_name="Top_Performers", index=False)

            bottom_df = pd.DataFrame(report["bottom_performers"])
            bottom_df.to_excel(writer, sheet_name="Bottom_Performers", index=False)

        logger.info(f"Excel report saved to: {excel_file}")

    def create_text_summary(self, report: Dict, summary_file: Path):
        """Create a human-readable text summary"""
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("INTERVIEW EVALUATION COMPREHENSIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(
                f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total Evaluations: {report['total_evaluations']}\n")
            f.write(f"Unique Questions: {report['unique_questions']}\n")
            f.write(f"Unique Domains: {report['unique_domains']}\n\n")

            # Overall statistics
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            basic_stats = report["basic_statistics"]["overall_score"]
            f.write(f"Average Score: {basic_stats['mean']:.2f}\n")
            f.write(f"Standard Deviation: {basic_stats['std']:.2f}\n")
            f.write(f"Minimum Score: {basic_stats['min']:.2f}\n")
            f.write(f"Maximum Score: {basic_stats['max']:.2f}\n\n")

            # Grade distribution
            f.write("GRADE DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            for grade, count in sorted(report["grade_distribution"].items()):
                percentage = (count / report["total_evaluations"]) * 100
                f.write(f"Grade {grade}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Domain performance
            f.write("PERFORMANCE BY DOMAIN\n")
            f.write("-" * 25 + "\n")
            domain_means = report["domain_analysis"]["overall_score"]["mean"]
            for domain, score in sorted(
                domain_means.items(), key=lambda x: x[1], reverse=True
            ):
                f.write(f"{domain}: {score:.2f}\n")

        logger.info(f"Summary report saved to: {summary_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive interview evaluation"
    )
    parser.add_argument(
        "--input", default=EVALUATION_CONFIG["input_file"], help="Input JSON file path"
    )
    parser.add_argument(
        "--config", default="evaluation_config.py", help="Configuration file path"
    )
    parser.add_argument(
        "--output",
        default=EVALUATION_CONFIG["output_directory"],
        help="Output directory path",
    )

    args = parser.parse_args()

    # Update config with command line arguments
    config = EVALUATION_CONFIG.copy()
    config["output_directory"] = args.output

    try:
        # Initialize and run evaluator
        evaluator = ComprehensiveEvaluator(config)
        result_file = evaluator.run_complete_evaluation(args.input)

        if result_file:
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved to: {evaluator.output_dir}")
            print(f"Main results file: {result_file}")
        else:
            print("Evaluation failed. Check logs for details.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
