import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluate import evaluate_batch, find_latest_batch_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PUBLIC_PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR_NAME = "plots"

CATEGORY_LABELS = {
    "Mathematical/Logical Reasoning": "Math",
    "Physics & Scientific Reasoning": "Physics",
    "Logic Puzzles & Constraint Satisfaction": "Logic",
    "Strategic Game Theory": "Game Theory",
}


def load_evaluation(batch_dir: Path) -> dict:
    metrics_path = batch_dir / "evaluation_metrics.json"
    if not metrics_path.exists():
        return evaluate_batch(batch_dir)
    with open(metrics_path, encoding="utf-8") as file:
        return json.load(file)


def load_baseline_metrics(batch_dir: Path) -> Optional[dict]:
    baseline_path = batch_dir / "baseline_metrics.json"
    if not baseline_path.exists():
        return None
    with open(baseline_path, encoding="utf-8") as file:
        return json.load(file)


def _percent_label(value: float) -> str:
    return f"{value * 100:.1f}%"


def _add_bar_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            _percent_label(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_overall_metrics(metrics: dict, output_dir: Path) -> Path:
    labels = [
        "Overall\nAccuracy",
        "Consensus\nRate",
        "Improvement\nRate",
        "Judge Accuracy\n(on Disagreement)",
    ]
    values = [
        metrics["overall_accuracy"],
        metrics["consensus_rate"],
        metrics["improvement_rate"],
        metrics["judge_accuracy_on_disagreement"] or 0.0,
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title("Debate System — Key Evaluation Metrics")
    _add_bar_labels(ax, bars)
    fig.tight_layout()

    path = output_dir / "overall_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy_by_category(metrics: dict, output_dir: Path) -> Path:
    categories = metrics["accuracy_by_category"]
    labels = [CATEGORY_LABELS.get(name, name) for name in categories]
    values = [stats["accuracy"] for stats in categories.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Answer Accuracy by Category")
    _add_bar_labels(ax, bars)
    fig.tight_layout()

    path = output_dir / "accuracy_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_solver_stage_comparison(metrics: dict, output_dir: Path) -> Path:
    solvers = list(metrics["stage_1_solver_accuracy"].keys())
    stage_1 = [metrics["stage_1_solver_accuracy"][s]["accuracy"] or 0 for s in solvers]
    stage_3 = [metrics["stage_3_solver_accuracy"][s]["accuracy"] or 0 for s in solvers]

    x_positions = range(len(solvers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_1 = ax.bar(
        [x - width / 2 for x in x_positions],
        stage_1,
        width,
        label="Stage 1 (Initial)",
        color="#F58518",
    )
    bars_3 = ax.bar(
        [x + width / 2 for x in x_positions],
        stage_3,
        width,
        label="Stage 3 (Refined)",
        color="#54A24B",
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(solvers)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Solver Accuracy Before vs After Refinement")
    ax.legend()
    _add_bar_labels(ax, bars_1)
    _add_bar_labels(ax, bars_3)
    fig.tight_layout()

    path = output_dir / "solver_stage_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_system_comparison(
    debate_accuracy: float,
    baselines: Optional[dict],
    output_dir: Path,
) -> Path:
    systems = ["Full Debate"]
    values = [baselines.get("full_debate", {}).get("accuracy", debate_accuracy) if baselines else debate_accuracy]

    if baselines:
        if "single_llm" in baselines:
            systems.append("Single LLM")
            values.append(baselines["single_llm"]["accuracy"])
        if "majority_vote" in baselines:
            systems.append("Majority Vote")
            values.append(baselines["majority_vote"]["accuracy"])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(systems)]
    bars = ax.bar(systems, values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("System Comparison")
    _add_bar_labels(ax, bars)
    fig.tight_layout()

    path = output_dir / "system_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_plots(batch_dir: Path) -> Dict[str, str]:
    sns.set_theme(style="whitegrid")
    evaluation = load_evaluation(batch_dir)
    metrics = evaluation["metrics"]
    baselines = load_baseline_metrics(batch_dir)

    output_dir = batch_dir / PLOTS_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {
        "overall_metrics": str(plot_overall_metrics(metrics, output_dir)),
        "accuracy_by_category": str(plot_accuracy_by_category(metrics, output_dir)),
        "solver_stage_comparison": str(plot_solver_stage_comparison(metrics, output_dir)),
        "system_comparison": str(plot_system_comparison(metrics["overall_accuracy"], baselines, output_dir)),
    }

    PUBLIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in saved.values():
        shutil.copy2(path, PUBLIC_PLOTS_DIR / Path(path).name)

    saved["public_plots_dir"] = str(PUBLIC_PLOTS_DIR)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation plots from batch results.")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Path to batch results directory (default: latest in results/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_dir = args.batch_dir or find_latest_batch_dir()
    if batch_dir is None or not batch_dir.exists():
        raise SystemExit("No batch results found. Run python -m src.run_batch first.")

    plot_paths = generate_plots(batch_dir)
    print("\n=== PLOTS GENERATED ===\n")
    for name, path in plot_paths.items():
        if name == "public_plots_dir":
            print(f"\nEasy-access copies saved to: {path}")
            continue
        print(f"{name}: {path}")
