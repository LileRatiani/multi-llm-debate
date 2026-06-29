import json
import re
from fractions import Fraction
from math import isclose
from pathlib import Path
from typing import List, Optional, Set

from sympy import simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.json"

SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE),
    re.compile(r"(?:exact\s+)?(?:final\s+)?(?:radius|answer|result)\s*(?:is|=)\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:final\s+)?answer\s*(?:is)?\s*[:=]\s*(.+)", re.IGNORECASE),
    re.compile(r"therefore[,]?\s*(?:the answer is\s*)?(.+)", re.IGNORECASE),
]

ANSWER_SYNONYMS = {
    "defect": ["defect", "take", "stop", "exit", "terminate", "grab", "end the game"],
}

UNIT_SUFFIX_PATTERN = re.compile(
    r"\s*(?:meters?|m/s|c|Ω|ohms?|degrees?|°|minutes?|mins?|gold coins?)\s*\.?$",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    """Normalize text for exact comparison."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("$", "")
    text = text.replace("°", " degrees")
    text = text.replace("\u202f", " ")
    text = text.replace("\u2248", " approx ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?]+$", "", text).strip()
    return text


def normalize_math_notation(value: str) -> str:
    """Convert common math/LaTeX notation into sympy-friendly text."""
    text = str(value).strip()
    text = text.split("≈")[0].split("\\approx")[0].strip()
    text = text.replace("√", "sqrt")
    text = re.sub(r"(\d+)\s*/\s*sqrt", r"\1/sqrt", text)
    text = re.sub(r"\^\{([^}]+)\}", r"**(\1)", text)
    text = re.sub(r"\{([^}]+)\}", r"\1", text)
    text = text.replace("^", "**")
    text = re.sub(r"\bT_f\b", "T", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_units_and_decorations(value: str) -> str:
    """Remove labels, units, and approximate parentheticals."""
    text = str(value).strip()
    text = re.sub(r"\([^)]*(?:approx|≈|~)[^)]*\)", "", text, flags=re.IGNORECASE)
    text = UNIT_SUFFIX_PATTERN.sub("", text).strip()

    if "=" in text:
        left, right = text.split("=", 1)
        if len(left) <= 30 and not re.search(r"[<>]", left):
            text = right.strip()

    return text.strip(" .")


def extract_answer(candidate: str) -> str:
    """Pull the most likely final answer from a verbose model response."""
    if candidate is None:
        return ""

    text = str(candidate).strip()
    if not text:
        return ""

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines[-1]

    return text


def generate_math_candidates(predicted: str) -> List[str]:
    """Build math-expression variants without lowercasing symbols."""
    candidates: Set[str] = set()
    raw_values = [str(predicted or "").strip(), extract_answer(predicted)]

    for value in raw_values:
        if not value:
            continue

        variants = [
            value,
            normalize_math_notation(value),
            strip_units_and_decorations(value),
            strip_units_and_decorations(normalize_math_notation(value)),
        ]

        for variant in variants:
            cleaned = variant.strip()
            if cleaned:
                candidates.add(cleaned)

    return list(candidates)


def generate_text_candidates(predicted: str) -> List[str]:
    """Build normalized text variants for exact matching."""
    return [normalize_text(candidate) for candidate in generate_math_candidates(predicted) if candidate]


def _try_fraction(value: str) -> Optional[Fraction]:
    value = value.strip()
    if re.fullmatch(r"-?\d+/\d+", value):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _try_float(value: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.\-eE]", "", value.strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_symbolic_expression(expression: str):
    expr = normalize_math_notation(expression)
    expr = strip_units_and_decorations(expr)
    expr = re.sub(r"\bsqrt\b", "sqrt", expr)
    return parse_expr(expr, transformations=SYMPY_TRANSFORMATIONS, evaluate=True)


def numeric_equivalent(predicted: str, ground_truth: str) -> bool:
    """Check numeric or simple fractional equivalence."""
    pred_fraction = _try_fraction(predicted)
    truth_fraction = _try_fraction(ground_truth)
    if pred_fraction is not None and truth_fraction is not None:
        return pred_fraction == truth_fraction

    pred_float = _try_float(predicted)
    truth_float = _try_float(ground_truth)
    if pred_float is not None and truth_float is not None:
        return isclose(pred_float, truth_float, rel_tol=1e-6, abs_tol=1e-9)

    return False


def contains_ground_truth_number(predicted: str, ground_truth: str) -> bool:
    """Match numeric ground truths embedded in verbose answers."""
    truth = ground_truth.strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", truth):
        return False

    numbers = re.findall(r"-?\d+(?:\.\d+)?", predicted)
    return truth in numbers or f"{truth}.0" in numbers


def synonym_match(predicted: str, ground_truth: str) -> bool:
    """Match semantically equivalent short answers such as Defect vs take."""
    truth = normalize_text(ground_truth)
    predicted_norm = normalize_text(predicted)

    synonyms = ANSWER_SYNONYMS.get(truth, [])
    return any(synonym in predicted_norm for synonym in synonyms)


def symbolic_equivalent(predicted: str, ground_truth: str) -> bool:
    """Check whether two mathematical expressions are equivalent."""
    ground_truth = ground_truth.strip()

    if normalize_text(predicted) == normalize_text(ground_truth):
        return True

    for candidate in generate_math_candidates(predicted):
        if normalize_text(candidate) == normalize_text(ground_truth):
            return True
        if numeric_equivalent(candidate, ground_truth):
            return True
        try:
            pred_expr = _parse_symbolic_expression(candidate)
            truth_expr = _parse_symbolic_expression(ground_truth)
            if simplify(pred_expr - truth_expr) == 0:
                return True
        except Exception:
            continue

    return False


def check_exact_match(predicted: str, ground_truth: str) -> bool:
    """Compare answers after normalization and light answer extraction."""
    truth = normalize_text(ground_truth)
    candidates = generate_text_candidates(predicted)

    if truth in candidates:
        return True

    for candidate in candidates:
        if candidate and truth in candidate:
            return True

    if contains_ground_truth_number(predicted, ground_truth):
        return True

    if synonym_match(predicted, ground_truth):
        return True

    return numeric_equivalent(extract_answer(predicted), ground_truth)


def check_answer(predicted: str, ground_truth: str, evaluation_type: str = "exact_match") -> dict:
    """
    Compare a model answer against the dataset ground truth.

    Returns a result dict with correctness and normalized values.
    """
    extracted = extract_answer(predicted)

    if evaluation_type == "symbolic_match":
        correct = symbolic_equivalent(predicted, ground_truth)
    else:
        correct = check_exact_match(predicted, ground_truth)

    return {
        "correct": correct,
        "evaluation_type": evaluation_type,
        "predicted_raw": predicted,
        "predicted_extracted": extracted,
        "ground_truth": ground_truth,
    }


def load_dataset(dataset_path: Path = DATASET_PATH) -> list:
    with open(dataset_path, encoding="utf-8") as file:
        return json.load(file)


def check_problem_answer(problem: dict, predicted: str) -> dict:
    """Check an answer for a dataset problem entry."""
    result = check_answer(
        predicted=predicted,
        ground_truth=problem["ground_truth_answer"],
        evaluation_type=problem.get("evaluation_type", "exact_match"),
    )
    result["problem_id"] = problem.get("problem_id")
    return result


def rescore_batch_directory(batch_dir: Path) -> dict:
    """Re-evaluate saved batch result files with the current checker."""
    dataset = {problem["problem_id"]: problem for problem in load_dataset()}
    results = []
    correct_count = 0

    for result_file in sorted(batch_dir.glob("*.json")):
        if result_file.name == "summary.json":
            continue

        with open(result_file, encoding="utf-8") as file:
            record = json.load(file)

        problem_id = record.get("problem_id")
        problem = dataset.get(problem_id)
        if not problem:
            continue

        check = check_problem_answer(problem, record.get("final_answer") or "")
        record["correct"] = check["correct"]
        record["answer_check"] = check

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)

        results.append(
            {
                "problem_id": problem_id,
                "correct": check["correct"],
                "final_answer": record.get("final_answer"),
                "ground_truth": problem["ground_truth_answer"],
            }
        )
        if check["correct"]:
            correct_count += 1

    summary_path = batch_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)
        summary["correct_count"] = correct_count
        summary["problems"] = results
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

    return {
        "total": len(results),
        "correct": correct_count,
        "accuracy": correct_count / len(results) if results else 0.0,
        "problems": results,
    }


if __name__ == "__main__":
    dataset = load_dataset()

    samples = [
        ("153", "153", "exact_match", True),
        ("The final answer is 153.", "153", "exact_match", True),
        ("0.5", "1/2", "symbolic_match", True),
        ("15/14", "15/14", "symbolic_match", True),
        ("T*(1/2)**(2/5)", "T*(1/2)^(2/5)", "symbolic_match", True),
        ("T_f = T * 2^{-2/5} ≈ 0.76 T", "T*(1/2)^(2/5)", "symbolic_match", True),
        ("9/sqrt(7)", "9/sqrt(7)", "symbolic_match", True),
        ("Exact radius = 9/√7 m (≈3.40 m).", "9/sqrt(7)", "symbolic_match", True),
        ("A is a knave, B is a knight, C is a knave", "A is a knave, B is a knight, C is a knave", "exact_match", True),
        ("2 stones from pile 3", "2 stones from pile 3", "exact_match", True),
        ("A can keep 98 gold coins in his optimal proposal.", "98", "exact_match", True),
        (
            "The first player should take (i.e., stop the game) at the very first decision node.",
            "Defect",
            "exact_match",
            True,
        ),
        ("≈ 0.7735 Ω", "0.5", "exact_match", False),
        ("38", "153", "exact_match", False),
    ]

    print("Running answer checker self-tests...\n")
    for predicted, ground_truth, evaluation_type, expected in samples:
        result = check_answer(predicted, ground_truth, evaluation_type)
        status = "PASS" if result["correct"] == expected else "FAIL"
        print(f"[{status}] {evaluation_type}: {predicted!r} vs {ground_truth!r} -> {result['correct']}")

    batch_dir = Path(__file__).resolve().parent.parent / "results" / "batch_20260628_173933"
    if batch_dir.exists():
        print("\nRe-scoring saved batch results...")
        rescored = rescore_batch_directory(batch_dir)
        print(f"Updated accuracy: {rescored['correct']}/{rescored['total']} ({rescored['accuracy']:.1%})")

    print(f"\nDataset loaded: {len(dataset)} problems")
