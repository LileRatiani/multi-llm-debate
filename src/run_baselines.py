import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.answer_checker import check_problem_answer, extract_answer, load_dataset, normalize_text
from src.config import Config
from src.evaluate import find_latest_batch_dir
from src.llm_client import generate_json_response
from src.orchestrator import InitialSolutionSchema

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SINGLE_LLM_MODEL = Config.SOLVER_1_MODEL
VOTING_MODELS = [
    ("Solver_1", Config.SOLVER_1_MODEL),
    ("Solver_2", Config.SOLVER_2_MODEL),
    ("Solver_3", Config.SOLVER_3_MODEL),
]

SYSTEM_PROMPT = (
    "You are an expert academic solver. "
    "Solve the problem with absolute technical accuracy. "
    "Break down your solution step-by-step. "
    "Keep thought_process concise (under 400 words) while remaining complete."
)


def create_baseline_dir(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            return json.load(file)
    return {
        "baseline_type": output_dir.name.split("_")[0],
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


def request_solution(model: str, question: str, temperature: float = 0.5) -> Optional[dict]:
    raw_response = generate_json_response(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Problem: {question}",
        model_name=model,
        response_schema=InitialSolutionSchema,
        temperature=temperature,
    )

    try:
        parsed = json.loads(raw_response) if raw_response.strip() else {}
        if not parsed.get("final_answer"):
            return None
        return parsed
    except json.JSONDecodeError:
        return None


def vote_key(answer: str) -> str:
    return normalize_text(extract_answer(answer))


def majority_vote(answers: Dict[str, str]) -> tuple:
    """Return (winning_answer, vote_counts_by_solver)."""
    buckets: Dict[str, List[str]] = {}
    for solver_name, answer in answers.items():
        key = vote_key(answer)
        buckets.setdefault(key, []).append(answer)

    winning_key, _ = Counter(
        {key: len(values) for key, values in buckets.items()}
    ).most_common(1)[0]

    return buckets[winning_key][0], {solver: vote_key(ans) for solver, ans in answers.items()}


def run_single_llm_baseline(
    output_dir: Path,
    dataset: List[dict],
    resume: bool = False,
    delay_seconds: float = 3.0,
) -> dict:
    print(f"\n=== SINGLE-LLM BASELINE ({SINGLE_LLM_MODEL}) ===")
    print(f"Saving results to: {output_dir}\n")

    summary = load_summary(output_dir)
    summary["baseline_type"] = "single_llm"
    summary["model"] = SINGLE_LLM_MODEL
    summary["total_problems"] = len(dataset)

    for index, problem in enumerate(dataset, start=1):
        problem_id = problem["problem_id"]
        result_path = output_dir / f"{problem_id}.json"

        if resume and result_path.exists():
            print(f"[{index}/{len(dataset)}] Skipping {problem_id} (already completed)")
            continue

        print(f"[{index}/{len(dataset)}] {problem_id} — single LLM")

        started_at = datetime.now(timezone.utc).isoformat()
        solution = request_solution(SINGLE_LLM_MODEL, problem["question"])
        final_answer = solution.get("final_answer") if solution else None
        check = check_problem_answer(problem, final_answer or "")

        record = {
            "problem_id": problem_id,
            "category": problem["category"],
            "question": problem["question"],
            "ground_truth": problem["ground_truth_answer"],
            "evaluation_type": problem["evaluation_type"],
            "model": SINGLE_LLM_MODEL,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if solution else "error",
            "solution": solution,
            "final_answer": final_answer,
            "correct": check["correct"],
            "answer_check": check,
        }

        label = "CORRECT" if check["correct"] else "WRONG"
        print(f"  -> {label}: {final_answer!r} (expected {problem['ground_truth_answer']!r})")

        save_problem_result(output_dir, problem_id, record)
        summary["problems"] = [p for p in summary["problems"] if p["problem_id"] != problem_id]
        summary["problems"].append(
            {
                "problem_id": problem_id,
                "correct": check["correct"],
                "final_answer": final_answer,
                "ground_truth": problem["ground_truth_answer"],
            }
        )
        summary["completed_count"] = len(summary["problems"])
        summary["correct_count"] = sum(1 for entry in summary["problems"] if entry["correct"])
        save_summary(output_dir, summary)

        if index < len(dataset) and delay_seconds > 0:
            time.sleep(delay_seconds)

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_summary(output_dir, summary)
    return summary


def run_voting_baseline(
    output_dir: Path,
    dataset: List[dict],
    resume: bool = False,
    delay_seconds: float = 5.0,
) -> dict:
    print(f"\n=== MAJORITY-VOTE BASELINE ===")
    print(f"Saving results to: {output_dir}\n")

    summary = load_summary(output_dir)
    summary["baseline_type"] = "majority_vote"
    summary["models"] = {name: model for name, model in VOTING_MODELS}
    summary["total_problems"] = len(dataset)

    for index, problem in enumerate(dataset, start=1):
        problem_id = problem["problem_id"]
        result_path = output_dir / f"{problem_id}.json"

        if resume and result_path.exists():
            print(f"[{index}/{len(dataset)}] Skipping {problem_id} (already completed)")
            continue

        print(f"[{index}/{len(dataset)}] {problem_id} — majority vote")

        started_at = datetime.now(timezone.utc).isoformat()
        solver_answers = {}
        solver_solutions = {}

        for solver_name, model in VOTING_MODELS:
            print(f"  -> {solver_name} ({model})")
            solution = request_solution(model, problem["question"])
            if solution:
                solver_solutions[solver_name] = solution
                solver_answers[solver_name] = solution["final_answer"]

        if len(solver_answers) < 2:
            final_answer = next(iter(solver_answers.values()), None)
            vote_details = solver_answers
        else:
            final_answer, vote_details = majority_vote(solver_answers)

        check = check_problem_answer(problem, final_answer or "")
        record = {
            "problem_id": problem_id,
            "category": problem["category"],
            "question": problem["question"],
            "ground_truth": problem["ground_truth_answer"],
            "evaluation_type": problem["evaluation_type"],
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if solver_answers else "error",
            "solver_solutions": solver_solutions,
            "solver_answers": solver_answers,
            "vote_details": vote_details,
            "final_answer": final_answer,
            "correct": check["correct"],
            "answer_check": check,
        }

        label = "CORRECT" if check["correct"] else "WRONG"
        print(f"  -> Vote winner: {final_answer!r} — {label}")

        save_problem_result(output_dir, problem_id, record)
        summary["problems"] = [p for p in summary["problems"] if p["problem_id"] != problem_id]
        summary["problems"].append(
            {
                "problem_id": problem_id,
                "correct": check["correct"],
                "final_answer": final_answer,
                "ground_truth": problem["ground_truth_answer"],
            }
        )
        summary["completed_count"] = len(summary["problems"])
        summary["correct_count"] = sum(1 for entry in summary["problems"] if entry["correct"])
        save_summary(output_dir, summary)

        if index < len(dataset) and delay_seconds > 0:
            time.sleep(delay_seconds)

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_summary(output_dir, summary)
    return summary


def summary_to_metrics(summary: dict) -> dict:
    total = summary.get("completed_count", 0)
    correct = summary.get("correct_count", 0)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "output_dir": str(summary.get("output_dir", "")),
    }


def write_baseline_metrics(
    debate_batch_dir: Path,
    single_summary: dict,
    voting_summary: dict,
) -> Path:
    debate_metrics_path = debate_batch_dir / "evaluation_metrics.json"
    debate_accuracy = None
    if debate_metrics_path.exists():
        with open(debate_metrics_path, encoding="utf-8") as file:
            evaluation = json.load(file)
        metrics = evaluation["metrics"]
        debate_accuracy = {
            "accuracy": metrics["overall_accuracy"],
            "correct": int(metrics["overall_accuracy_count"].split("/")[0]),
            "total": metrics["total_problems"],
        }

    payload = {
        "single_llm": summary_to_metrics(single_summary),
        "majority_vote": summary_to_metrics(voting_summary),
    }
    if debate_accuracy:
        payload["full_debate"] = debate_accuracy

    output_path = debate_batch_dir / "baseline_metrics.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-LLM and majority-vote baselines.")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Debate batch directory to attach baseline_metrics.json to",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "single", "voting"],
        default="both",
        help="Which baseline(s) to run",
    )
    parser.add_argument("--single-dir", type=Path, default=None, help="Output dir for single-LLM results")
    parser.add_argument("--voting-dir", type=Path, default=None, help="Output dir for voting results")
    parser.add_argument("--resume", action="store_true", help="Skip completed problems")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N problems")
    parser.add_argument("--delay-single", type=float, default=3.0, help="Delay between single-LLM problems")
    parser.add_argument("--delay-voting", type=float, default=5.0, help="Delay between voting problems")
    parser.add_argument("--regen-plots", action="store_true", help="Regenerate plots after baselines finish")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    debate_batch_dir = args.batch_dir or find_latest_batch_dir()
    if debate_batch_dir is None:
        raise SystemExit("No debate batch directory found.")

    dataset = load_dataset()
    if args.limit is not None:
        dataset = dataset[: args.limit]

    single_dir = args.single_dir or create_baseline_dir("baselines_single_llm")
    voting_dir = args.voting_dir or create_baseline_dir("baselines_voting")
    single_dir.mkdir(parents=True, exist_ok=True)
    voting_dir.mkdir(parents=True, exist_ok=True)

    single_summary = {"completed_count": 0, "correct_count": 0, "problems": []}
    voting_summary = {"completed_count": 0, "correct_count": 0, "problems": []}

    if args.mode in ("both", "single"):
        single_summary = run_single_llm_baseline(
            output_dir=single_dir,
            dataset=dataset,
            resume=args.resume,
            delay_seconds=args.delay_single,
        )
        single_summary["output_dir"] = str(single_dir)

    if args.mode in ("both", "voting"):
        voting_summary = run_voting_baseline(
            output_dir=voting_dir,
            dataset=dataset,
            resume=args.resume,
            delay_seconds=args.delay_voting,
        )
        voting_summary["output_dir"] = str(voting_dir)

    metrics_path = write_baseline_metrics(debate_batch_dir, single_summary, voting_summary)

    print(f"\n{'=' * 60}")
    print("BASELINES COMPLETE")
    if args.mode in ("both", "single"):
        print(
            f"Single LLM:     {single_summary['correct_count']}/{single_summary['completed_count']} "
            f"({single_summary['correct_count'] / single_summary['completed_count']:.1%})"
            if single_summary["completed_count"]
            else "Single LLM: no results"
        )
    if args.mode in ("both", "voting"):
        print(
            f"Majority vote:  {voting_summary['correct_count']}/{voting_summary['completed_count']} "
            f"({voting_summary['correct_count'] / voting_summary['completed_count']:.1%})"
            if voting_summary["completed_count"]
            else "Majority vote: no results"
        )
    print(f"Metrics saved to: {metrics_path}")
    print(f"{'=' * 60}")

    if args.regen_plots:
        from src.plot_results import generate_plots

        generate_plots(debate_batch_dir)
        print(f"Plots regenerated for: {debate_batch_dir}")
