# HMITL: Manager-Governed LLM Iteration with Guardrails and Rollback for Reproducible Healthcare Machine Learning Pipelines

**Jialuo Chen · Haijing Wang · Siyu Shao**

**2026 IEEE 14th International Conference on Healthcare Informatics (ICHI)**  
Pages 1218–1226

[📄 IEEE Xplore](https://ieeexplore.ieee.org/document/11634837) ·
[🔗 DOI](https://doi.org/10.1109/ICHI69079.2026.00156) ·
[💻 Code](https://github.com/chenj926/ICHI_AgentDS_clai) ·
[🤗 AgentDS-Healthcare](https://huggingface.co/datasets/lainmn/AgentDS-Healthcare) ·
[📚 Citation](#citation)

> **TL;DR —** HMITL is a manager-governed human–LLM collaboration protocol for iterative healthcare machine learning. Instead of allowing an LLM to autonomously modify a pipeline, the human manager controls evidence admission, data-integrity guardrails, deterministic evaluation, acceptance, and rollback. The protocol is evaluated across three multimodal AgentDS-Healthcare tasks spanning structured data, clinical text, PDF receipts, and time-series records.

---

## Overview

Large language models can rapidly generate and modify machine-learning pipelines, but unconstrained iteration can introduce silent data errors, unstable feature engineering, unnecessary complexity, and performance regressions.

**Human-Manager-in-the-Loop (HMITL)** separates proposal generation from acceptance:

- **Builder LLM** proposes concrete pipeline changes.
- **Human Manager** maintains the task brief, guardrails, and acceptance criteria.
- **Evaluation Harness** deterministically evaluates every candidate.
- **Rollback** restores the best-known pipeline after regressions.
- **Artifact Logging** preserves prompts, decisions, metrics, and experiment traces for auditing and reproducibility.

The repository accompanies our paper published at **IEEE ICHI 2026** and contains the code, notebooks, experiment traces, and reproducibility artifacts used in the study.

<!-- Recommended:
<p align="center">
  <img src="docs/assets/hmitl_overview.png" width="850">
</p>
-->

---

## Key Results

| Task | Metric | Naive Baseline | HMITL Result | Setting |
|---|---:|---:|---:|---|
| Challenge 1 — 30-day readmission | Macro-F1 ↑ | 0.6198 | **0.9014** | Competition window |
| Challenge 2 — ED cost forecasting | MAE ↓ | 701.1945 | **447.9542** | Post-competition, official evaluation interface |
| Challenge 3 — discharge readiness | Macro-F1 ↑ | 0.4889 | **0.8408** | Controlled 15-iteration study |

For Challenge 3, the multi-window arbitration + rollback configuration achieved the highest score while also producing the smallest maximum drawdown and adjacent-iteration variance among the evaluated variants.

---

## Reviewer / Reader Quick Start

If you only want the shortest paper-relevant path through the repository:

1. Read the **Highlights** section below.
2. Create a clean Python environment and install `requirements.txt`.
3. Copy `.env.example` to `.env` and set your local paths and benchmark credentials.
4. Download the upstream **AgentDS-Healthcare** data.
5. Point `CLAI_BASE_DIR` to your local Healthcare data directory.
6. Go directly to `agent_ds_healthcare/`.
7. Use the canonical best-entry notebooks:
   - **Challenge 1**: `Challenge1_Health_Final.ipynb`
   - **Challenge 2**: `Challenge2_baseline_ichi_best.ipynb`
   - **Challenge 3**: `Challenge3_ichi_best.ipynb`
8. Treat `ch2_artifacts/` and `ch3_artifacts/` as **audit logs and reproducibility artifacts**. They are included for transparency; reviewers are not expected to inspect every artifact file line by line.

---

## Highlights

### Paper framing

This repository documents a manager-governed approach to LLM-assisted data-science iteration under explicit guardrails, deterministic evaluation, and rollback-aware selection.

### Main benchmark tasks covered in this release

- **Challenge 1** - 30-day readmission prediction
- **Challenge 2** - ED cost forecasting
- **Challenge 3** - discharge readiness prediction

### Paper-relevant best public entry points

| Benchmark task | Canonical notebook | Notes |
|---|---|---|
| Challenge 1 | `agent_ds_healthcare/Challenge1_Health_Final.ipynb` | Final public Challenge 1 notebook retained in working form |
| Challenge 2 | `agent_ds_healthcare/Challenge2_baseline_ichi_best.ipynb` | Separate best-entry notebook added for easier reviewer navigation |
| Challenge 3 | `agent_ds_healthcare/Challenge3_ichi_best.ipynb` | Separate best-entry notebook added for easier reviewer navigation |

If you want the paper process rather than only the best final entry points, also inspect the archived experiment notebooks and artifact folders under `agent_ds_healthcare/`.

---

## What Is in This Repository

### Paper-relevant content

- `agent_ds_healthcare/`
  - Challenge 1, 2, and 3 notebooks
  - public best-entry notebooks for Challenge 2 and Challenge 3
  - reproducibility artifacts and experiment logs
- `agent_ds_healthcare/ch2_artifacts/`
  - Challenge 2 manager logs, iteration summaries, optimization traces, and supporting artifacts
- `agent_ds_healthcare/ch3_artifacts/`
  - Challenge 3 JSONL traces and controlled-study records

### Additional non-core content

- `agent_ds_commerce/`
  - retained from broader AgentDS participation
  - **not required** to review or reproduce the healthcare paper results

---

## Repository Layout

```text
.
├── agent_ds_healthcare/
│   ├── Challenge1_Health_Final.ipynb
│   ├── Challenge2_baseline_ichi_best.ipynb
│   ├── Challenge3_ichi_best.ipynb
│   ├── ch2_artifacts/
│   ├── ch3_artifacts/
│   └── ...
├── agent_ds_commerce/                  # auxiliary; not needed for the healthcare paper
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tested Environment

This public release was prepared around:

- **Python 3.12**
- Windows local development for the author-side runs
- notebook-based execution for the paper path

A clean virtual environment is strongly recommended.

---

## Installation

### 1. Create a clean virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Register the environment as a notebook kernel

```bash
python -m ipykernel install --user --name hmitl-agentds --display-name "Python (hmitl-agentds)"
```

### 3. GPU note for PyTorch users

If you want GPU-enabled local runs, install the PyTorch wheel that matches your CUDA setup first, then install the rest of the requirements. CPU-only installation is sufficient for basic repository inspection and many reproduction checks.

---

## Configuration via `.env`

Create a local `.env` file in the repository root by copying `.env.example`:

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

#### macOS / Linux

```bash
cp .env.example .env
```

Then edit `.env` and fill in your values:

```env
AGENTDS_API_KEY=your_agentds_api_key_here
AGENTDS_TEAM_NAME=your_team_name_here
CLAI_BASE_DIR=D:/AgentDs/agent_ds_healthcare
CLAI_RECEIPT_DIR=D:/AgentDs/agent_ds_healthcare/receipt
```

### What the variables mean

- `AGENTDS_API_KEY`: your benchmark submission API key
- `AGENTDS_TEAM_NAME`: your exact AgentDS team name
- `CLAI_BASE_DIR`: root folder that contains the Healthcare benchmark files
- `CLAI_RECEIPT_DIR`: optional override for the raw receipt PDF folder used in Challenge 2
- `CLAI_BRANCH`, `FEATURE_CACHE_MODE`, `FORCE_RETRAIN`: optional experiment controls used by some legacy notebooks

### Important

Do **not** commit your real `.env` file or benchmark credentials.

---

## AgentDS-Healthcare Data

This repository does **not** redistribute the raw benchmark data or the raw receipt PDFs. The dataset is publicly available upstream and should be downloaded directly from the official AgentDS-Healthcare source.

### Recommended download method on Windows

Because the benchmark includes many files and a large receipt-PDF folder, Windows users are encouraged to use **Git Bash** or the **Hugging Face CLI** rather than manually downloading files one by one.

### Option A: Hugging Face CLI snapshot download

```bash
pip install -U huggingface_hub hf-xet
huggingface-cli download lainmn/AgentDS-Healthcare --repo-type dataset --local-dir ./data/AgentDS-Healthcare
```

### Option B: Git / Git Bash clone

```bash
git lfs install
git clone https://huggingface.co/datasets/lainmn/AgentDS-Healthcare ./data/AgentDS-Healthcare
```

After download, the Healthcare files will typically live under:

```text
./data/AgentDS-Healthcare/Healthcare
```

Point `CLAI_BASE_DIR` in your `.env` file to that directory.

---

## Expected Local Data Layout

### Recommended external layout (author-style Windows path)

```text
D:/AgentDs/agent_ds_healthcare/
├── admissions_train.csv
├── admissions_test.csv
├── discharge_notes.json
├── ed_cost_train.csv
├── ed_cost_test.csv
├── patients.csv
├── stays_train.csv
├── stays_test.csv
├── vitals_timeseries.json
├── receipts_pdf/             # upstream naming; either is fine
│   ├── receipt_<patient_id>.pdf
│   └── ...
└── receipts_parsed.joblib    # derived cache used by the Challenge 2 best notebook
```

### Alternative repo-local layout

You may also keep the benchmark files directly under a repository-local folder:

```text
./agent_ds_healthcare/
```

and set:

```env
CLAI_BASE_DIR=./agent_ds_healthcare
```

### Directory naming note for Challenge 2 receipts

The upstream dataset uses `receipts_pdf/`. In the author's local workflow, the raw receipt PDFs may also be stored under `receipt/`.

Either layout is acceptable as long as:

- the raw PDF filenames remain unchanged, and
- `CLAI_RECEIPT_DIR` points to the correct folder when needed.

---

## Challenge 2 Receipt Cache

The best public Challenge 2 notebook expects a derived cache file:

```text
<CLAI_BASE_DIR>/receipts_parsed.joblib
```

This cache is **not** committed to the repository and should be generated locally from the raw receipt PDFs before running the best Challenge 2 notebook.

In other words:

- raw PDFs stay in `receipt/` or `receipts_pdf/`
- the derived parsed cache should be saved as `receipts_parsed.joblib` under `CLAI_BASE_DIR`

If you do not already have that cache, prepare it once locally before running the Challenge 2 best notebook.

---

## Path Handling and Legacy Notebook Notes

This repository contains both cleaned entry-point notebooks and historical working notebooks.

### Recommended path convention

Use `.env` plus `CLAI_BASE_DIR` as the single source of truth for your data location.

### If your local directory is different

That is completely fine.

For example, if your data lives at:

```text
E:/datasets/AgentDS/Healthcare
```

set:

```env
CLAI_BASE_DIR=E:/datasets/AgentDS/Healthcare
```

### Legacy notebook note

Some newer notebooks are already closer to environment-variable-based configuration, while some legacy notebooks still contain a small hard-coded path block near the top. If a notebook does not yet read `CLAI_BASE_DIR` directly, update **only that path configuration block** to your local data directory rather than editing the entire notebook.

---

## How to Run the Main Notebooks

## 1) Challenge 1 - 30-day readmission

Canonical entry point:

```text
agent_ds_healthcare/Challenge1_Health_Final.ipynb
```

Expected core inputs:

- `admissions_train.csv`
- `admissions_test.csv`
- `patients.csv`
- `discharge_notes.json`

Expected submission format:

```text
admission_id,readmit_30d
```

---

## 2) Challenge 2 - ED cost forecasting

Canonical entry point:

```text
agent_ds_healthcare/Challenge2_baseline_ichi_best.ipynb
```

Expected core inputs:

- `ed_cost_train.csv`
- `ed_cost_test.csv`
- `patients.csv`
- `admissions_train.csv`
- `admissions_test.csv`
- `receipts_parsed.joblib`

Raw receipt PDFs should remain available locally under either `receipt/` or `receipts_pdf/` if you need to rebuild the parsed cache.

Expected submission format:

```text
patient_id,ed_cost_next3y_usd
```

---

## 3) Challenge 3 - discharge readiness prediction

Canonical entry point:

```text
agent_ds_healthcare/Challenge3_ichi_best.ipynb
```

Expected core inputs:

- `stays_train.csv`
- `stays_test.csv`
- `patients.csv`
- `vitals_timeseries.json`

Expected submission format:

```text
stay_id,discharge_ready_day11
```

---

## Recommended Credential Pattern Inside Notebooks

For a public release, benchmark credentials should not be hard-coded in notebooks. The recommended pattern is:

```python
import os
from dotenv import load_dotenv
from agentds import BenchmarkClient

load_dotenv()

client = BenchmarkClient(
    api_key=os.environ["AGENTDS_API_KEY"],
    team_name=os.environ["AGENTDS_TEAM_NAME"],
)
```

Similarly, use `CLAI_BASE_DIR` to define the local dataset path rather than hard-coding a machine-specific directory.

---

## Reproducibility Artifacts

### `ch2_artifacts/`

Challenge 2 artifacts include manager-governed iteration logs, optimization traces, supporting notes, and reproducibility records.

### `ch3_artifacts/`

Challenge 3 artifacts include JSONL traces, controlled-study records, and rollback-related evidence.

These files are included so that the **decision process** is inspectable, not only the final leaderboard-facing outputs.

---

## Notes for Reviewers

1. **The main paper path is under `agent_ds_healthcare/`.**
2. **You are not expected to inspect every artifact file.** The artifact folders are included for auditability.
3. **Some artifacts preserve raw working records.** A small subset may contain concise prompt fragments, iteration notes, or partially bilingual working material. They are preserved for transparency and are not required for understanding the main paper claims.
4. **The best-entry notebooks are separated for convenience.** Historical notebooks are retained because the paper is about both performance and the process that produced it.
5. **Challenge 2 requires a local parsed receipt cache.** The raw PDFs are upstream; the derived `receipts_parsed.joblib` should be prepared locally.

---

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@misc{chen2026hmitl,
  title={HMITL: Manager-Governed LLM Iteration with Guardrails and Rollback for Reproducible Healthcare Machine Learning Pipelines},
  author={Jialuo Chen and Haijing Wang and Siyu Shao},
  year={2026}
}
```

Please also cite the upstream AgentDS benchmark and respect the dataset license and usage terms.

---

## Acknowledgements

We thank the AgentDS Benchmark team for releasing AgentDS-Healthcare and the challenge infrastructure that made this study possible.

---

## Upstream References

- AgentDS-Healthcare dataset: <https://huggingface.co/datasets/lainmn/AgentDS-Healthcare>
- AgentDS benchmark: <https://huggingface.co/datasets/lainmn/AgentDS>
- AgentDS website: <https://agentds.org>
