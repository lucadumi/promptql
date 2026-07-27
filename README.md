# Small LLM: Fine-tune (or Pre-train), then Evaluate

Train a small language model I actually control, then measure it honestly. This maps
almost line-for-line onto the Mistral Applied Scientist / Research Engineer JD, which
lists pre-training, post-training, deployment on GPU, OOM/NCCL troubleshooting, data
curation, and evaluation.

## Why this project
- **Mistral (both intern roles)**: the closest possible proof that I can do the actual
  job. It turns "interested in LLMs" into "I trained and evaluated one."
- **Perplexity, ASML, ZEISS, TomTom**: shows real PyTorch model work, not just API calls.
- Fills the same CV gap as the forecasting project (a genuine train-and-evaluate story)
  but on the LLM side.

## Pick one track (Track A is recommended to start)
### Track A - LoRA fine-tune a small open model (recommended)
Fine-tune a 1B to 3B open model (e.g. a small Llama/Qwen/Mistral open checkpoint) on a
focused dataset using LoRA/QLoRA. Cheap, fits on a free/low-cost GPU, and directly
exercises post-training + evaluation.

### Track B - Pre-train a tiny model from scratch (nanoGPT style)
Train a small GPT from scratch on a compact corpus (e.g. TinyStories or a code subset).
More about the pre-training loop and optimization; great for the "trained from scratch"
story and understanding what actually happens inside.

Doing A first, then a small B, tells a complete pre-training + post-training story.

## Tech stack
- Python, PyTorch.
- Hugging Face `transformers`, `datasets`, `peft` (LoRA), `accelerate`,
  `bitsandbytes` (4-bit for QLoRA).
- Experiment tracking: Weights & Biases or plain CSV + matplotlib.
- Compute: Google Colab / Kaggle (free T4), Lightning Studios, or a cheap rented GPU
  (vast.ai / RunPod) when I need more VRAM.

## Repo layout & quickstart
```
src/eval_baseline.py   # Week 0: run the base model on the eval set, save before-numbers
src/data_utils.py      # shared DB schema + prompt format (keep train == eval)
src/metrics.py         # SQL normalisation + exact-match scoring
data/eval/             # held-out text-to-SQL eval set (20 examples) + data card
results/               # baseline_*.json + baseline.md leaderboard (committed)
requirements.in        # local (CPU/MPS) deps; requirements-gpu.txt = CUDA-only extras
Makefile               # make setup / make smoke / make baseline
```

Local setup (Apple Silicon, CPU/MPS):
```bash
make setup        # venv + install torch/transformers/datasets/peft/accelerate
make smoke        # fast pipeline check (tiny model, 5 examples)
make baseline     # real baseline: Qwen2.5-0.5B-Instruct on the 20-example eval set
```
Heavy training (LoRA/QLoRA) belongs on a **cloud T4** (Colab/Kaggle), not this Mac:
`bitsandbytes` / 4-bit is CUDA-only. Take `requirements-gpu.txt` to the GPU box.

## Milestones (target: 3 to 4 weeks, part-time)
- [ ] **Week 0 - Setup & baseline eval.** Env + repo. Load the base model, run it on a
      small eval set, and record baseline numbers *before* any training. You cannot
      show improvement without a before. _(scaffold + eval script ready — run `make baseline`.)_
- [ ] **Week 1 - Data curation.** Pick and clean a focused dataset (instruction pairs,
      a domain corpus, or a task like SQL generation). Document how I cleaned and split
      it. Data quality is half the JD.
- [ ] **Week 2 - Fine-tune (LoRA).** Get a training run to converge. Deliberately hit
      and then fix an out-of-memory error (gradient checkpointing, batch size, 4-bit),
      because "troubleshoot OOM" is literally in the JD. Log loss curves.
- [ ] **Week 3 - Evaluate.** Compare fine-tuned vs base on the same eval set, both
      quantitatively (task metric or a small benchmark) and with qualitative examples.
      Be honest about regressions.
- [ ] **Week 4 - Writeup + optional deploy.** README with method, curves, before/after
      table, and failure cases. Optional: quantize and serve it behind a tiny API.

## Evaluation
- A clear **before vs after** table on a held-out eval set.
- One task metric (e.g. exact-match / pass rate) plus 3 to 5 qualitative examples.
- Note what got worse, not just what got better. Honesty reads as maturity.

## Pitfalls to avoid (these become interview stories)
- No baseline eval before training (then you cannot prove anything).
- Overfitting a tiny dataset. Watch eval loss, not just train loss.
- Tokenization/prompt-format mistakes that silently wreck results.
- Claiming success from vibes. Always show the eval set.

## Stretch goals
- QLoRA on a bigger base model to practice memory tricks.
- Quantize (GGUF / bitsandbytes) and measure quality vs latency tradeoff.
- Combine with my existing RAG work (BoonBrain) for a retrieval + fine-tuned-model demo.
- A tiny distributed run (2 GPUs) to touch the NCCL / multi-GPU world the JD mentions.

## Deliverables
- Public GitHub repo + README (portfolio link for applications).
- Training curves, before/after eval table, and sample generations committed.
- A short "what I learned" section (especially the OOM/eval war stories).

## Do this next (first concrete actions)
1. `cd ~/Documents/small-llm-finetuning && python3 -m venv .venv && source .venv/bin/activate`
2. `pip install torch transformers datasets peft accelerate bitsandbytes && pip freeze > requirements.txt`
3. `git init` and first commit.
4. Choose Track A and pick a small base model + a focused dataset (start narrow, e.g.
   natural-language-to-SQL or a single instruction style).
5. Before training anything, run the base model on 20 eval examples and save the results.

## How to talk about it in applications
"I fine-tuned a small open model with LoRA, hit an OOM wall, fixed it with gradient
checkpointing and 4-bit, and then proved it beat the base model on a held-out eval set,
while being honest about where it regressed." That sentence hits four separate lines of
the Mistral JD.
