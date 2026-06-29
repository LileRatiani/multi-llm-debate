import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.answer_checker import check_problem_answer, extract_answer, normalize_text

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SOLVER_NAMES = ("Solver_1", "Solver_2", "Solver_3")


def find_latest_batch_dir(results_dir: Path = RESULTS_DIR) -> Optional[Path]:
    batch_dirs = sorted(results_dir.glob("batch_*"), key=lambda path: path.name, reverse=True)
    return batch_dirs[0] if batch_dirs else None


SKIP_BATCH_FILES = {"summary.json", "evaluation_metrics.json"}


def load_batch_records(batch_dir: Path) -> List[dict]:
    records = []
    for result_file in sorted(batch_dir.glob("*.json")):
        if result_file.name in SKIP_BATCH_FILES:
            continue
        with open(result_file, encoding="utf-8") as file:
            record = json.load(file)
        if "problem_id" not in record:
            continue
        records.append(record)
    return records


def problem_metadata(record: dict) -> dict:
    return {
        "problem_id": record["problem_id"],
        "category": record.get("category"),
        "ground_truth_answer": record["ground_truth"],
        "evaluation_type": record.get("evaluation_type", "exact_match"),
    }


def normalize_answer_key(answer: Optional[str]) -> str:
    return normalize_text(extract_answer(answer or ""))


def is_correct(problem: dict, answer: Optional[str]) -> bool:
    if not answer:
        return False
    return check_problem_answer(problem, answer)["correct"]


def solver_stage_answers(debate_result: dict) -> Dict[str, Dict[str, Optional[str]]]:
    stage_1 = debate_result.get("stage_1") or {}
    stage_3 = debate_result.get("stage_3") or {}

    answers = {}
    for solver in SOLVER_NAMES:
        initial = stage_1.get(solver) or {}
        refined = stage_3.get(solver) or {}
        answers[solver] = {
            "stage_1": initial.get("final_answer"),
            "stage_3": refined.get("refined_answer") if refined else None,
        }
    return answers


def evaluate_record(record: dict) -> dict:
    problem = problem_metadata(record)
    debate = record.get("debate_result") or {}
    stage_4 = debate.get("stage_4") or {}
    judgment = stage_4.get("judgment") or {}
    winner = judgment.get("winner")

    solver_answers = solver_stage_answers(debate)
    stage_1_keys = []
    stage_1_correct = {}
    stage_3_correct = {}
    solver_improved = {}

    for solver, answers in solver_answers.items():
        if answers["stage_1"] is not None:
            stage_1_keys.append(normalize_answer_key(answers["stage_1"]))
            stage_1_correct[solver] = is_correct(problem, answers["stage_1"])

        if answers["stage_3"] is not None:
            stage_3_correct[solver] = is_correct(problem, answers["stage_3"])
            if solver in stage_1_correct:
                solver_improved[solver] = (
                    not stage_1_correct[solver] and stage_3_correct[solver]
                )

    valid_stage_1 = [key for key in stage_1_keys if key]
    consensus = len(valid_stage_1) >= 2 and len(set(valid_stage_1)) == 1
    disagreement = len(valid_stage_1) >= 2 and len(set(valid_stage_1)) > 1

    final_correct = is_correct(problem, record.get("final_answer"))
    winner_correct = False
    if winner:
        winner_refined = solver_answers.get(winner, {}).get("stage_3")
        winner_initial = solver_answers.get(winner, {}).get("stage_1")
        if winner_refined is not None:
            winner_correct = is_correct(problem, winner_refined)
        elif winner_initial is not None:
            winner_correct = is_correct(problem, winner_initial)

    winner_improved = bool(winner and solver_improved.get(winner, False))
    any_solver_improved = any(solver_improved.values())

    return {
        "problem_id": problem["problem_id"],
        "category": problem["category"],
        "final_correct": final_correct,
        "consensus": consensus,
        "disagreement": disagreement,
        "winner": winner,
        "winner_correct": winner_correct,
        "winner_improved": winner_improved,
        "any_solver_improved": any_solver_improved,
        "stage_1_correct": stage_1_correct,
        "stage_3_correct": stage_3_correct,
        "solver_improved": solver_improved,
        "stage_1_answers": {
            solver: solver_answers[solver]["stage_1"] for solver in SOLVER_NAMES
        },
    }


def aggregate_metrics(per_problem: List[dict]) -> dict:
    total = len(per_problem)
    if total == 0:
        return {"total_problems": 0}

    final_correct = sum(1 for item in per_problem if item["final_correct"])
    consensus_count = sum(1 for item in per_problem if item["consensus"])
    improved_count = sum(1 for item in per_problem if item["any_solver_improved"])
    winner_improved_count = sum(1 for item in per_problem if item["winner_improved"])

    disagreement_cases = [item for item in per_problem if item["disagreement"]]
    judge_correct_on_disagreement = sum(
        1 for item in disagreement_cases if item["winner_correct"]
    )

    by_category = defaultdict(lambda: {"total": 0, "correct": 0})
    for item in per_problem:
        category = item["category"] or "Unknown"
        by_category[category]["total"] += 1
        if item["final_correct"]:
            by_category[category]["correct"] += 1

    stage_1_solver_totals = {solver: {"correct": 0, "total": 0} for solver in SOLVER_NAMES}
    stage_3_solver_totals = {solver: {"correct": 0, "total": 0} for solver in SOLVER_NAMES}
    for item in per_problem:
        for solver in SOLVER_NAMES:
            if solver in item["stage_1_correct"]:
                stage_1_solver_totals[solver]["total"] += 1
                if item["stage_1_correct"][solver]:
                    stage_1_solver_totals[solver]["correct"] += 1
            if solver in item["stage_3_correct"]:
                stage_3_solver_totals[solver]["total"] += 1
                if item["stage_3_correct"][solver]:
                    stage_3_solver_totals[solver]["correct"] += 1

    return {
        "total_problems": total,
        "overall_accuracy": final_correct / total,
        "overall_accuracy_count": f"{final_correct}/{total}",
        "consensus_rate": consensus_count / total,
        "consensus_count": f"{consensus_count}/{total}",
        "improvement_rate": improved_count / total,
        "improvement_count": f"{improved_count}/{total}",
        "winner_improvement_rate": winner_improved_count / total,
        "winner_improvement_count": f"{winner_improved_count}/{total}",
        "judge_accuracy_on_disagreement": (
            judge_correct_on_disagreement / len(disagreement_cases)
            if disagreement_cases
            else None
        ),
        "judge_accuracy_on_disagreement_count": (
            f"{judge_correct_on_disagreement}/{len(disagreement_cases)}"
            if disagreement_cases
            else "N/A"
        ),
        "disagreement_cases": len(disagreement_cases),
        "accuracy_by_category": {
            category: {
                "accuracy": stats["correct"] / stats["total"],
                "count": f"{stats['correct']}/{stats['total']}",
            }
            for category, stats in sorted(by_category.items())
        },
        "stage_1_solver_accuracy": {
            solver: {
                "accuracy": (
                    stage_1_solver_totals[solver]["correct"] / stage_1_solver_totals[solver]["total"]
                    if stage_1_solver_totals[solver]["total"]
                    else None
                ),
                "count": (
                    f"{stage_1_solver_totals[solver]['correct']}/"
                    f"{stage_1_solver_totals[solver]['total']}"
                ),
            }
            for solver in SOLVER_NAMES
        },
        "stage_3_solver_accuracy": {
            solver: {
                "accuracy": (
                    stage_3_solver_totals[solver]["correct"] / stage_3_solver_totals[solver]["total"]
                    if stage_3_solver_totals[solver]["total"]
                    else None
                ),
                "count": (
                    f"{stage_3_solver_totals[solver]['correct']}/"
                    f"{stage_3_solver_totals[solver]['total']}"
                ),
            }
            for solver in SOLVER_NAMES
        },
    }


def print_report(metrics: dict, per_problem: List[dict]) -> None:
    print("\n=== DEBATE SYSTEM EVALUATION ===\n")
    print(f"Problems evaluated: {metrics['total_problems']}")
    print(f"Overall accuracy:          {metrics['overall_accuracy']:.1%} ({metrics['overall_accuracy_count']})")
    print(f"Consensus rate (Stage 1):  {metrics['consensus_rate']:.1%} ({metrics['consensus_count']})")
    print(f"Improvement rate (Stage 3): {metrics['improvement_rate']:.1%} ({metrics['improvement_count']})")
    print(f"Winner improved (Stage 3): {metrics['winner_improvement_rate']:.1%} ({metrics['winner_improvement_count']})")

    judge_accuracy = metrics["judge_accuracy_on_disagreement"]
    if judge_accuracy is None:
        print("Judge accuracy (disagreement): N/A (no disagreement cases)")
    else:
        print(
            "Judge accuracy (disagreement): "
            f"{judge_accuracy:.1%} ({metrics['judge_accuracy_on_disagreement_count']})"
        )

    print("\nAccuracy by category:")
    for category, stats in metrics["accuracy_by_category"].items():
        print(f"  - {category}: {stats['accuracy']:.1%} ({stats['count']})")

    print("\nStage 1 solver accuracy:")
    for solver, stats in metrics["stage_1_solver_accuracy"].items():
        if stats["accuracy"] is None:
            print(f"  - {solver}: N/A")
        else:
            print(f"  - {solver}: {stats['accuracy']:.1%} ({stats['count']})")

    print("\nStage 3 solver accuracy:")
    for solver, stats in metrics["stage_3_solver_accuracy"].items():
        if stats["accuracy"] is None:
            print(f"  - {solver}: N/A")
        else:
            print(f"  - {solver}: {stats['accuracy']:.1%} ({stats['count']})")

    incorrect = [item for item in per_problem if not item["final_correct"]]
    if incorrect:
        print("\nIncorrect final answers:")
        for item in incorrect:
            print(f"  - {item['problem_id']}")

    print()


def evaluate_batch(batch_dir: Path, save: bool = True) -> dict:
    records = load_batch_records(batch_dir)
    per_problem = [evaluate_record(record) for record in records]
    metrics = aggregate_metrics(per_problem)

    output = {
        "batch_dir": str(batch_dir),
        "metrics": metrics,
        "per_problem": per_problem,
    }

    if save:
        output_path = batch_dir / "evaluation_metrics.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved debate batch results.")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Path to a batch results directory (default: latest in results/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_dir = args.batch_dir or find_latest_batch_dir()
    if batch_dir is None or not batch_dir.exists():
        raise SystemExit("No batch results found. Run python -m src.run_batch first.")

    results = evaluate_batch(batch_dir)
    print_report(results["metrics"], results["per_problem"])
    print(f"Saved metrics to: {batch_dir / 'evaluation_metrics.json'}")
