# Small-LLM Fine-Tuning: Natural Language → SQL

> Take a small, **open** language model I fully control, teach it to translate plain-English
> questions into SQL, and **prove** it improved with an honest before/after evaluation.

**Status:** ✅ Week 0 complete — baseline recorded.
The base model (`Qwen2.5-0.5B-Instruct`) scores **40% exact-match** on a held-out set
*before any fine-tuning*. The goal of the project is to beat that number with LoRA and
measure the gain honestly.

---

## What this project is

A complete **post-training + evaluation loop** in miniature:

1. Pick a small open model I can run and modify myself.
2. **Measure it first** (the "before").
3. **Fine-tune** it on a narrow task (text-to-SQL) using LoRA.
4. **Re-measure** on the *same* held-out set (the "after"), and be honest about what got
   worse, not just what got better.

It's deliberately small and cheap so the whole loop runs on a laptop + a free cloud GPU —
the point is the *method*, not the model size.

---

## The task: text-to-SQL

Given a **fixed database schema** and a question in English, the model must output a
single SQL query. The schema is shown to the model in every prompt:

```sql
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER, location TEXT);
CREATE TABLE employees  (id INTEGER PRIMARY KEY, name TEXT, department TEXT,
                         salary INTEGER, hire_date TEXT, manager_id INTEGER);
```

| Input (question)                 | Target (SQL)                          |
|----------------------------------|---------------------------------------|
| "How many employees are there?"  | `SELECT COUNT(*) FROM employees`      |
| "List all department names."     | `SELECT name FROM departments`        |

---

## Results

Held-out eval set: **20 examples** (`data/eval/text2sql_eval.jsonl`).

| Model                                    | Params | Metric       | Score        |
|------------------------------------------|:------:|--------------|:------------:|
| `Qwen2.5-0.5B-Instruct` (base, no FT)    | 0.49B  | exact-match  | **40% (8/20)** |
| `+ LoRA fine-tune` (Week 2)              |   —    | exact-match  | _TBD_        |

<sub>Baseline: greedy (deterministic) decoding, run on Apple Silicon GPU (MPS) in ~11s.
Full per-example predictions in `results/`.</sub>

**Read the score correctly:** this is **strict normalized exact-match** — a
semantically-correct query that differs by a single keyword (e.g. `ORDER BY salary` vs
`ORDER BY salary ASC`) is still counted as *wrong*. So 40% is a conservative **lower
bound**; it can never inflate the "before". Moving to *execution accuracy* is a planned
upgrade (see [Method](#how-it-works-method)).

---

## Quickstart

**Requirements:** Python 3.9+, macOS or Linux. On Apple Silicon the Mac GPU (MPS) is used
automatically; no NVIDIA card needed to *run* the baseline.

```bash
make setup      # create a venv and install torch / transformers / datasets / peft / accelerate
make baseline   # reproduce the 40% baseline (downloads Qwen2.5-0.5B-Instruct once)
make smoke      # fast end-to-end pipeline check on a tiny model (no big download)
```

Try any other small model:

```bash
python -m src.eval_baseline --model Qwen/Qwen2.5-1.5B-Instruct
python -m src.eval_baseline --model HuggingFaceTB/SmolLM2-360M-Instruct --limit 10
```

---

## How it works (method)

- **Prompt format** — schema + question, rendered with the model's own chat template.
  The *exact same* format is used for training and evaluation to avoid prompt-format
  drift (a classic silent-failure bug). → `src/data_utils.py`
- **Generation** — greedy / deterministic (`do_sample=False`), so every run is
  reproducible and before/after comparisons are fair. → `src/eval_baseline.py`
- **Scoring** — normalize both prediction and gold (strip markdown fences, lowercase,
  collapse whitespace, unify quotes, drop trailing `;`), then compare as strings.
  → `src/metrics.py`
- **Output** — a machine-readable `results/baseline_*.json` (every prediction) plus a
  human-readable `results/baseline.md` leaderboard.
- **Planned upgrade** — *execution accuracy*: run predicted and gold SQL against a real
  SQLite database and compare the returned rows. This credits correct-but-differently-
  written queries and is the honest way to measure the fine-tuned model in Week 3.

---

## Repo layout

```
.
├── src/
│   ├── eval_baseline.py   # run a model on the eval set, save the before/after numbers
│   ├── data_utils.py      # DB schema + shared prompt format (keeps train == eval)
│   └── metrics.py         # SQL normalisation + exact-match scoring
├── data/eval/
│   ├── text2sql_eval.jsonl  # 20 held-out NL→SQL examples
│   └── README.md            # data card: schema + how the set was built
├── results/
│   ├── baseline.md          # human-readable results table
│   └── baseline_*.json      # full per-example predictions (committed)
├── requirements.in          # local (CPU/MPS) top-level deps
├── requirements.txt         # pinned lockfile (pip freeze)
├── requirements-gpu.txt     # CUDA-only extras (bitsandbytes) for the GPU box
└── Makefile                 # make setup / smoke / baseline
```

---

## Roadmap

- [x] **Week 0 — Setup & baseline.** Env, repo, eval set, and the base model's
      before-numbers (**40%**). *You cannot show improvement without a before.*
- [ ] **Week 1 — Data curation.** Build a separate NL→SQL **training** set (no leakage
      into eval), document cleaning and splits.
- [ ] **Week 2 — Fine-tune (LoRA).** Get a run to converge on a cloud GPU; deliberately
      hit and then fix an out-of-memory error (gradient checkpointing, batch size, 4-bit).
      Log loss curves.
- [ ] **Week 3 — Evaluate.** Fine-tuned vs base on the same eval set, quantitatively
      (exact-match → execution accuracy) and with qualitative examples. Report regressions.
- [ ] **Week 4 — Writeup + optional deploy.** Method, curves, before/after table, failure
      cases. Optional: quantize (GGUF / bitsandbytes) and serve behind a tiny API.

---

## Compute & environment

- **Local (this repo):** Apple Silicon macOS, CPU/MPS — used for scaffolding and light
  evaluation.
- **Training:** a cloud **T4** (Google Colab / Kaggle, free tier). `bitsandbytes` / 4-bit
  QLoRA is **CUDA-only**, so it lives in `requirements-gpu.txt` and is *not* installed
  locally.

## Tech stack

Python · PyTorch · Hugging Face `transformers`, `datasets`, `peft` (LoRA), `accelerate`
(+ `bitsandbytes` on the GPU). Experiment tracking: JSON/CSV now, Weights & Biases optional.

## Scope (tracks)

- **Track A — LoRA fine-tune a small open model (current).** Post-training + evaluation on
  a focused task. Cheap, fits a free GPU.
- **Track B — pre-train a tiny GPT from scratch (stretch, nanoGPT-style).** For the
  "trained from scratch" story and to understand the pre-training loop itself.

---

## Notes on honesty (things that become good interview stories)

- Baseline measured **before** training — otherwise you can prove nothing. ✅ done.
- Watch **eval** loss, not just train loss (a tiny dataset overfits fast).
- Keep tokenization / prompt format **identical** between train and eval.
- Always show the eval set — no success-by-vibes.

## What I learned

_(Filled in as the project progresses — especially the OOM and evaluation war stories.)_

---

## Motivation

A portfolio project for **AI / ML research & engineering internships**. It demonstrates the
real end-to-end workflow those roles describe — data curation, post-training with LoRA, GPU
training (including OOM troubleshooting), and honest evaluation — rather than just calling a
hosted API.

> One-line pitch: *"I fine-tuned a small open model with LoRA, hit and fixed an OOM wall,
> and proved it beat the base model on a held-out eval set — while being honest about where
> it regressed."*
