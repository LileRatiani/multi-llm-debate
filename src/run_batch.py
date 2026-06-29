import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.answer_checker import check_problem_answer, load_dataset
from src.orchestrator import DebateOrchestrator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def create_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            return json.load(file)
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "total_problems": 0,
        "completed_count": 0,
        "correct_count": 0,
        "problems": [],
    }


def save_summary(output_dir: Path, summary: dict) -> None:
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def save_problem_result(output_dir: Path, problem_id: str, record: dict) -> None:
    with open(output_dir / f"{problem_id}.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)


def run_batch(
    output_dir: Path,
    resume: bool = False,
    limit: Optional[int] = None,
    delay_seconds: float = 10.0,
    problem_ids: Optional[List[str]] = None,
) -> dict:
    dataset = load_dataset()
    if problem_ids:
        dataset = [problem for problem in dataset if problem["problem_id"] in problem_ids]
    if limit is not None:
        dataset = dataset[:limit]

    orchestrator = DebateOrchestrator()
    summary = load_summary(output_dir)
    summary["total_problems"] = len(dataset)
    summary["output_dir"] = str(output_dir)

    print(f"Running batch debate on {len(dataset)} problems")
    print(f"Saving results to: {output_dir}\n")

    for index, problem in enumerate(dataset, start=1):
        problem_id = problem["problem_id"]
        result_path = output_dir / f"{problem_id}.json"

        if resume and result_path.exists():
            print(f"[{index}/{len(dataset)}] Skipping {problem_id} (already completed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{index}/{len(dataset)}] {problem_id} — {problem['category']}")
        print(f"{'=' * 60}")
        print(f"Question: {problem['question']}\n")

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            debate_result = orchestrator.run_full_debate(problem["question"])
            final_answer = debate_result.get("final_answer")
            check = check_problem_answer(problem, final_answer or "")

            record = {
                "problem_id": problem_id,
                "category": problem["category"],
                "question": problem["question"],
                "ground_truth": problem["ground_truth_answer"],
                "evaluation_type": problem["evaluation_type"],
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "final_answer": final_answer,
                "correct": check["correct"],
                "answer_check": check,
                "debate_result": debate_result,
            }

            status_label = "CORRECT" if check["correct"] else "WRONG"
            print(f"\n-> {problem_id}: {status_label}")
            print(f"   Predicted: {final_answer!r}")
            print(f"   Expected:  {problem['ground_truth_answer']!r}")

        except Exception as error:
            record = {
                "problem_id": problem_id,
                "category": problem["category"],
                "question": problem["question"],
                "ground_truth": problem["ground_truth_answer"],
                "evaluation_type": problem["evaluation_type"],
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": str(error),
                "final_answer": None,
                "correct": False,
            }
            print(f"\n-> {problem_id}: ERROR — {error}")

        save_problem_result(output_dir, problem_id, record)

        summary["problems"] = [
            entry for entry in summary["problems"] if entry["problem_id"] != problem_id
        ]
        summary["problems"].append(
            {
                "problem_id": problem_id,
                "status": record["status"],
                "correct": record.get("correct", False),
                "final_answer": record.get("final_answer"),
                "ground_truth": problem["ground_truth_answer"],
            }
        )
        summary["completed_count"] = len(summary["problems"])
        summary["correct_count"] = sum(1 for entry in summary["problems"] if entry.get("correct"))
        save_summary(output_dir, summary)

        if index < len(dataset) and delay_seconds > 0:
            print(f"\nWaiting {delay_seconds:.0f}s before next problem (rate limit buffer)...")
            time.sleep(delay_seconds)

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_summary(output_dir, summary)

    print(f"\n{'=' * 60}")
    print("BATCH COMPLETE")
    print(f"Problems run: {summary['completed_count']}/{summary['total_problems']}")
    print(f"Correct: {summary['correct_count']}/{summary['completed_count']}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full debate pipeline on the dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: results/batch_<timestamp>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip problems that already have result files in the output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N problems (useful for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Seconds to wait between problems (default: 10)",
    )
    parser.add_argument(
        "--problem-id",
        action="append",
        dest="problem_ids",
        help="Run only specific problem IDs (can be passed multiple times)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or create_output_dir(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_batch(
        output_dir=output_dir,
        resume=args.resume,
        limit=args.limit,
        delay_seconds=args.delay,
        problem_ids=args.problem_ids,
    )
