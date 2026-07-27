"""LoRA fine-tuning for the text-to-SQL task -- a transparent, hand-written loop.

This is the "tutoring" step: we take the base model that scored 40% on the sealed
eval exam and teach it on data/train/ with LoRA, then (separately) re-run the exam
to measure the honest before/after gain.

It is written as an explicit PyTorch loop (no HF Trainer) so every ML concept is
visible:
  * LoRA (peft)         -- freeze the 0.49B base weights, train ~1% of new params.
  * Prompt masking      -- compute the loss ONLY on the SQL answer tokens, never on
                           the schema/question (labels = -100 for prompt tokens).
                           This is THE detail that makes instruction fine-tuning work.
  * Gradient accumulation-- simulate a bigger batch than fits in memory.
  * Warmup + linear decay -- stabilise early steps, then anneal the learning rate.
  * Train vs. val loss  -- a tiny dataset overfits fast, so we watch BOTH.

Reuses the SAME prompt format as eval (src/data_utils.build_messages) so training
and evaluation are identical -- otherwise the before/after comparison is invalid.

Run it (from the repo root, inside the venv):
    python -m src.train_lora --smoke     # fast plumbing check (16 ex, 1 epoch)
    python -m src.train_lora             # real run (Qwen 0.5B, LoRA, ~3 epochs)

The trained LoRA adapter is written to adapters/ (gitignored). Evaluate it with
the companion flag on the eval script (added next):
    python -m src.eval_baseline --adapter adapters/lora-qwen2.5-0.5b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Make `src` importable whether run as `-m src.train_lora` or as a file path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import build_messages, build_plain_prompt, load_jsonl  # noqa: E402
from src.metrics import exact_match  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO_ROOT / "data" / "train" / "text2sql_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "data" / "train" / "text2sql_val.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "adapters" / "lora-qwen2.5-0.5b"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Qwen2.5 linear projection layers LoRA plugs adapters into (attention + MLP).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Turn one {question, sql} row into a supervised training example.
#
# input_ids  = [ ---- prompt (schema+question) ---- ][ --- gold SQL --- ][EOS]
# labels     = [ -100, -100, ................, -100 ][ gold SQL tokens  ][EOS]
#                \___ ignored by the loss (masked) __/\__ the only tokens we learn _/
#
# The model still *reads* the prompt (it's in input_ids); we just don't ask it to
# *predict* the prompt. It only gets graded on producing the SQL.
# ---------------------------------------------------------------------------
def build_supervised_example(tokenizer, question: str, sql: str, max_len: int) -> Dict[str, List[int]]:
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            build_messages(question), add_generation_prompt=True
        )
    else:
        prompt_ids = tokenizer(build_plain_prompt(question)).input_ids

    completion_ids = tokenizer(sql, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]  # teach it to STOP

    input_ids = (prompt_ids + completion_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch: List[Dict[str, List[int]]], pad_id: int):
    """Right-pad a batch to its longest sequence; pad labels with -100."""
    import torch

    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)          # padding is ignored too
        attention.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate(model, loader, device) -> float:
    """Mean cross-entropy on the validation split (no gradients)."""
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def load_base_and_tokenizer(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model + tokenizer: {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16 only pays off on CUDA; fp32 is the safe/stable default on CPU/MPS.
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    return model, tokenizer


def attach_lora(model, r: int, alpha: int, dropout: float):
    """Freeze the base weights and insert trainable LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # e.g. "trainable: 0.5M || all: 494M || 0.1%"
    return model


def sample_predictions(model, tokenizer, rows, device, n: int, max_new_tokens: int) -> None:
    """Qualitative sanity check: generate SQL for a few val questions."""
    from src.eval_baseline import generate_sql  # reuse the exact eval-time generation

    model.config.use_cache = True
    print("\n--- sample generations on held-out val ---")
    for ex in rows[:n]:
        pred = generate_sql(model, tokenizer, ex["question"], device, max_new_tokens)
        ok = exact_match(pred, ex["sql"])
        print(f"[{'OK' if ok else 'XX'}] Q: {ex['question']}")
        print(f"      gold: {ex['sql']}")
        print(f"      pred: {pred}")
    model.config.use_cache = False


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    parser = argparse.ArgumentParser(description="LoRA fine-tune for text-to-SQL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val-file", default=str(DEFAULT_VAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10, help="optimizer steps")
    parser.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
    parser.add_argument("--limit", type=int, default=None, help="cap #train examples")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--smoke", action="store_true", help="tiny fast run to test plumbing")
    args = parser.parse_args()

    if args.smoke:  # fast end-to-end plumbing check
        args.limit = args.limit or 16
        args.epochs = 1
        args.batch_size = 4
        args.eval_every = 5
        args.log_every = 2

    set_seed(args.seed)
    device = pick_device(args.device)

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    if args.limit is not None:
        train_rows = train_rows[: args.limit]

    print(f"Model: {args.model} | device: {device} | "
          f"train: {len(train_rows)} | val: {len(val_rows)}", flush=True)

    model, tokenizer = load_base_and_tokenizer(args.model, device)
    model = attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.config.use_cache = False   # not needed while training; silences a warning
    model.to(device)
    model.train()

    # Tokenise everything up front (the dataset is tiny).
    train_ds = [build_supervised_example(tokenizer, r["question"], r["sql"], args.max_seq_len)
                for r in train_rows]
    val_ds = [build_supervised_example(tokenizer, r["question"], r["sql"], args.max_seq_len)
              for r in val_rows]

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_id))

    # Optimiser sees only the LoRA params (everything else is frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    max_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * max_steps), max_steps
    )

    log: List[Dict] = []

    def record(step: int, epoch: int, split: str, loss: float) -> None:
        log.append({"step": step, "epoch": epoch, "split": split, "loss": round(loss, 5)})

    # Baseline val loss BEFORE any training (start of the loss curve).
    v0 = evaluate(model, val_loader, device)
    record(0, 0, "val", v0)
    print(f"[step 0] initial val_loss = {v0:.4f}")

    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / args.grad_accum).backward()   # scale for accumulation
            running += loss.item()
            seen += 1

            # One optimiser update per grad_accum micro-batches (or at epoch end).
            if i % args.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running / seen
                    running, seen = 0.0, 0
                    lr = scheduler.get_last_lr()[0]
                    print(f"epoch {epoch} | step {global_step}/{max_steps} "
                          f"| train_loss {avg:.4f} | lr {lr:.2e}", flush=True)
                    record(global_step, epoch, "train", avg)

                if global_step % args.eval_every == 0:
                    vl = evaluate(model, val_loader, device)
                    print(f"epoch {epoch} | step {global_step}/{max_steps} "
                          f"| val_loss {vl:.4f}", flush=True)
                    record(global_step, epoch, "val", vl)

        # End-of-epoch validation (always).
        vl = evaluate(model, val_loader, device)
        print(f"== end epoch {epoch}: val_loss {vl:.4f} ==", flush=True)
        record(global_step, epoch, "val", vl)

    elapsed = time.time() - t0

    # ---- save the adapter + a run record ----------------------------------
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)          # writes ONLY the small LoRA adapter
    tokenizer.save_pretrained(out_dir)

    trainable_n, all_n = model.get_nb_trainable_parameters()
    meta = {
        "base_model": args.model,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": LORA_TARGET_MODULES},
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "trainable_params": trainable_n,
        "all_params": all_n,
        "final_val_loss": log[-1]["loss"],
        "seconds": round(elapsed, 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (out_dir / "training_log.jsonl").open("w", encoding="utf-8") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")

    print("\n" + "=" * 60)
    print(f"DONE in {elapsed:.1f}s on {device}. "
          f"trainable={trainable_n:,} / {all_n:,} ({100*trainable_n/all_n:.2f}%)")
    try:
        shown_path = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        shown_path = out_dir
    print(f"Adapter saved to: {shown_path}")
    print("=" * 60)

    sample_predictions(model, tokenizer, val_rows, device, n=3, max_new_tokens=64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
