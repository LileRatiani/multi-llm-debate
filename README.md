# Multi-LLM Collaborative Debate System

A debate system where multiple LLMs solve problems independently, cross-evaluate each other's solutions, refine their answers based on peer feedback, and a final judge selects the best result.

## Overview

The pipeline uses **4 Groq models** with dynamic role assignment:

| Role | Default model |
|------|----------------|
| Solver pool | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gpt-oss-120b` |
| Judge pool | `gpt-oss-20b` |

**Pipeline stages:**
- **Stage 0** — Each LLM self-assesses which role fits best for the problem
- **Stage 0.5** — Deterministic algorithm assigns 3 Solvers + 1 Judge
- **Stage 1** — Independent solution generation
- **Stage 2** — Peer review (each solver reviews the other two)
- **Stage 3** — Refinement based on feedback
- **Stage 4** — Judge selects the winning answer

## Project Structure

```
multi_llm_debate/
├── data/
│   └── dataset.json          # 25 problems with ground-truth answers
├── plots/                    # Generated evaluation plots (easy access)
├── results/                  # Batch run outputs (gitignored)
├── src/
│   ├── orchestrator.py       # Full debate pipeline (Stages 0–4)
│   ├── llm_client.py         # Groq API client
│   ├── config.py             # Model configuration
│   ├── answer_checker.py     # Exact + symbolic answer matching
│   ├── run_batch.py          # Run debate on full dataset
│   ├── run_baselines.py      # Single-LLM + majority-vote baselines
│   ├── evaluate.py           # Compute evaluation metrics
│   └── plot_results.py       # Generate plots
├── requirements.txt
└── .env                      # Your API key (not committed)
```

## Setup

### 1. Clone and install dependencies

```bash
cd multi_llm_debate
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Verify API connection (optional)

```bash
python -m src.test_api
```

## Usage

All commands should be run from the `multi_llm_debate` directory.

### Run a single debate (demo)

```bash
python -m src.orchestrator
```

Runs the full pipeline on one hardcoded test problem and prints the final answer.

### Run debate on full dataset (25 problems)

```bash
python -m src.run_batch
```

Options:
```bash
python -m src.run_batch --limit 2          # Test on first 2 problems
python -m src.run_batch --resume           # Resume interrupted run
python -m src.run_batch --delay 5          # Seconds between problems
```

Results are saved to `results/batch_<timestamp>/`.

### Run baselines

```bash
python -m src.run_baselines --regen-plots
```

Runs:
- **Single-LLM** — one model, one shot per problem
- **Majority vote** — 3 models, pick most common answer

Options:
```bash
python -m src.run_baselines --limit 2
python -m src.run_baselines --mode single   # Only single-LLM
python -m src.run_baselines --mode voting   # Only majority vote
python -m src.run_baselines --resume --single-dir results/baselines_single_llm_<timestamp> --voting-dir results/baselines_voting_<timestamp>
```

### Evaluate results

```bash
python -m src.evaluate
```

Computes metrics and saves `evaluation_metrics.json` to the latest batch folder.

### Generate plots

```bash
python -m src.plot_results
```

Plots are saved to:
- `plots/` (project root, easy to find)
- `results/batch_<timestamp>/plots/`

### Test answer checker

```bash
python -m src.answer_checker
```

## Results (25-problem evaluation)

| Approach | Accuracy |
|----------|----------|
| **Full debate** | **96%** (24/25) |
| Majority vote | 56% (14/25) |
| Single LLM | 52% (13/25) |

**Debate system metrics:**
- Consensus rate (Stage 1): 20%
- Improvement rate (Stage 3): 40%
- Judge accuracy on disagreement: 95%

See `plots/system_comparison.png` for the visual comparison.

## Dataset

`data/dataset.json` contains 25 problems across 4 categories:
- Mathematical / Logical Reasoning (6)
- Physics & Scientific Reasoning (6)
- Logic Puzzles & Constraint Satisfaction (6)
- Strategic Game Theory (7)

Each entry includes `ground_truth_answer` and `evaluation_type` (`exact_match` or `symbolic_match`).

## Notes

- Full batch runs make **~15+ API calls per problem**. Use `--resume` if rate limits interrupt a run.
- The `results/` folder is gitignored. Copy `plots/` to the repo or regenerate with `python -m src.plot_results`.
- Models are configured in `src/config.py`.
