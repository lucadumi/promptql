"""Baseline evaluation: run a base model on the text-to-SQL eval set BEFORE any
training, and save the numbers + generations.

The README's #1 rule is: you cannot show improvement without a "before". This
script produces that "before":
  - loads a (small) base model + tokenizer,
  - generates a SQL query for each held-out question (greedy, deterministic),
  - scores exact-match against the gold SQL,
  - writes a JSON record (per-example + summary) and appends a Markdown row to
    results/baseline.md.

Run it (from the repo root, inside the venv):
    python -m src.eval_baseline --smoke --limit 5          # fast plumbing check
    python -m src.eval_baseline                            # real baseline (Qwen 0.5B)
    python -m src.eval_baseline --model <hf-model-id>      # any other small model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make `src` importable whether run as `-m src.eval_baseline` or as a file path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import SCHEMA_SQL, build_messages, build_plain_prompt, load_jsonl  # noqa: E402
from src.db import SCHEMAS, execution_match  # noqa: E402
from src.metrics import exact_match, normalize_sql  # noqa: E402
from src.repair import generate_with_repair, sqlite_validator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = REPO_ROOT / "data" / "eval" / "text2sql_eval.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "results"

# A tiny model used only to verify the pipeline end-to-end (produces nonsense SQL).
SMOKE_MODEL = "sshleifer/tiny-gpt2"
# A small, capable instruct model that runs on CPU/MPS for a real baseline.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device(pref: str) -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def quantize_dynamic_int8(model):
    """Apply PyTorch dynamic int8 quantization to a model's linear layers.

    "Dynamic" means the weights are quantized once, ahead of time, while
    activations are quantized on the fly per batch. It needs no calibration data
    and is the cheapest useful form of quantization -- which also makes it the
    honest one to report, since anything better would need a calibration set that
    this project does not have.

    CPU-only: PyTorch's dynamic quantization kernels do not run on MPS or CUDA.

    A backend has to be selected explicitly. On Apple Silicon PyTorch ships the
    `qnnpack` kernels but leaves `torch.backends.quantized.engine` set to "none",
    so quantizing without this raises a bare `NoQEngine` RuntimeError from deep
    inside `linear_prepack`, which is a genuinely confusing way to find out.
    """
    import torch

    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not engines:
        raise RuntimeError(
            "this PyTorch build has no int8 quantization backend "
            f"(supported_engines={torch.backends.quantized.supported_engines}); "
            "dynamic quantization is unavailable here"
        )
    if torch.backends.quantized.engine == "none":
        # fbgemm is the x86 kernel set, qnnpack the ARM one; prefer whichever
        # this build actually shipped rather than hardcoding a platform.
        torch.backends.quantized.engine = "fbgemm" if "fbgemm" in engines else engines[0]

    # LoRA adapters must be folded into the base weights first: quantizing a
    # PeftModel would leave the adapter layers in float and quantize around them,
    # which measures neither the adapter nor the quantization honestly.
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def load_model_and_tokenizer(model_name: str, device: str, adapter: str | None = None,
                             quantize: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer + model: {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float32 is the safe default on CPU/MPS; fp16 only really helps on CUDA.
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    # Load the LoRA adapter on top of the frozen base model, if one was given.
    # This is how we score the FINE-TUNED model on the exact same eval set --
    # same base, same prompt format, only the adapter weights differ.
    if adapter:
        from peft import PeftModel

        print(f"Applying LoRA adapter: {adapter} ...", flush=True)
        model = PeftModel.from_pretrained(model, adapter)

    if quantize:
        if device != "cpu":
            raise ValueError("--quantize requires --device cpu (PyTorch limitation)")
        print("Quantizing to int8 (dynamic, linear layers) ...", flush=True)
        model = quantize_dynamic_int8(model)

    model.to(device)
    model.eval()
    return model, tokenizer


def generate_sql(
    model, tokenizer, question: str, device: str, max_new_tokens: int,
    schema_sql: str = SCHEMA_SQL,
) -> str:
    import torch

    # Use the model's chat template when it has one; otherwise a plain prompt.
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            build_messages(question, schema_sql),
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(
            build_plain_prompt(question, schema_sql), return_tensors="pt"
        ).input_ids

    input_ids = input_ids.to(device)
    # batch size 1, no padding -> mask is all ones; passing it silences the
    # "attention mask not set" warning and keeps generation reliable.
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # greedy -> deterministic baseline
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline eval for the text-to-SQL task.")
    parser.add_argument("--model", default=None, help="HF model id (default: Qwen 0.5B Instruct).")
    parser.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter dir (score the fine-tuned model).")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny model to test the pipeline.")
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL), help="Path to eval JSONL.")
    parser.add_argument("--schema", default="employees", choices=sorted(SCHEMAS),
                        help="Which seeded schema the eval set targets (prompt + execution DB).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate per query.")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--quantize", action="store_true",
                        help="dynamic int8 quantization of the linear layers (CPU only)")
    parser.add_argument("--repair", type=int, default=1, metavar="N",
                        help="total generation attempts per question (1 = off). When a "
                             "generated query fails to execute, re-ask with the SQLite "
                             "error appended. Never sees the gold; see src/repair.py.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR), help="Where to write results.")
    args = parser.parse_args()

    model_name = args.model or (SMOKE_MODEL if args.smoke else DEFAULT_MODEL)
    device = pick_device(args.device)

    examples = load_jsonl(args.eval_file)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        print(f"No eval examples found in {args.eval_file}", file=sys.stderr)
        return 1

    label = model_name + (f" + {os.path.basename(os.path.normpath(args.adapter))}" if args.adapter else "")
    schema = SCHEMAS[args.schema]
    print(f"Model: {label} | device: {device} | examples: {len(examples)} | schema: {schema.name}", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_name, device, args.adapter,
                                                args.quantize)

    records = []
    correct = 0
    exec_correct = 0
    n_repaired = 0
    validate = sqlite_validator(schema)
    t0 = time.time()
    for i, ex in enumerate(examples, start=1):
        question, gold = ex["question"], ex["sql"]

        def _gen(q: str) -> str:
            return generate_sql(model, tokenizer, q, device, args.max_new_tokens,
                                schema.ddl)

        rep = generate_with_repair(_gen, question, validate, max_attempts=args.repair)
        raw = rep.raw
        n_repaired += int(rep.repaired)
        ok = exact_match(raw, gold)
        exec_res = execution_match(raw, gold, schema)
        correct += int(ok)
        exec_correct += int(exec_res.match)
        records.append(
            {
                "id": ex.get("id", i),
                "question": question,
                "gold": gold,
                "prediction_raw": raw,
                "prediction_norm": normalize_sql(raw),
                "gold_norm": normalize_sql(gold),
                "correct": ok,
                "exec_correct": exec_res.match,
                "exec_error": exec_res.pred_error,
                "attempts": rep.attempts,
                "repair_errors": rep.errors,
            }
        )
        flags = f"EM {'OK' if ok else 'XX'} | EX {'OK' if exec_res.match else 'XX'}"
        if rep.attempts > 1:
            flags += f" | {rep.attempts} tries"
        print(f"[{i:2d}/{len(examples)}] {flags}  {question}", flush=True)

    elapsed = time.time() - t0
    n = len(examples)
    em = correct / n
    ex_acc = exec_correct / n
    summary = {
        "model": model_name,
        "adapter": args.adapter,
        "schema": schema.name,
        "device": device,
        "quantized": bool(args.quantize),
        "repair_attempts": args.repair,
        "n_repaired": n_repaired,
        "eval_file": os.path.relpath(args.eval_file, REPO_ROOT),
        "n_examples": n,
        "exact_match": round(em, 4),
        "execution_accuracy": round(ex_acc, 4),
        "correct": correct,
        "exec_correct": exec_correct,
        "seconds": round(elapsed, 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_model = model_name.replace("/", "__")
    tag = safe_model
    if args.adapter:
        tag += "__" + os.path.basename(os.path.normpath(args.adapter))
    if args.quantize:
        # Without this an int8 run and an fp32 run of the same model differ only
        # by timestamp, and "latest file wins" silently mixes the two.
        tag += "__int8"
    if args.repair > 1:
        # Same reasoning: a repaired run is a different system, not a rerun.
        tag += f"__repair{args.repair}"
    tag += "__" + Path(args.eval_file).stem
    prefix = "eval" if args.adapter else "baseline"
    json_path = outdir / f"{prefix}_{tag}_{stamp}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=2)

    _append_markdown_row(outdir / "baseline.md", summary)

    print("\n" + "=" * 60)
    print(f"EXACT MATCH: {correct}/{n} = {em:.1%}")
    print(f"EXEC  ACC  : {exec_correct}/{n} = {ex_acc:.1%}  ({elapsed:.1f}s on {device})")
    if args.repair > 1:
        print(f"REPAIRED   : {n_repaired}/{n} queries needed a retry that then ran")
    print(f"Saved: {json_path.relative_to(REPO_ROOT)}")
    print("=" * 60)
    return 0


def _append_markdown_row(md_path: Path, summary: dict) -> None:
    """Maintain a human-readable results/baseline.md leaderboard."""
    header = (
        "| timestamp (UTC) | model | eval set | device | n | exact-match | exec-accuracy |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    display_model = summary["model"]
    if summary.get("adapter"):
        display_model += " + " + os.path.basename(os.path.normpath(summary["adapter"]))
    if summary.get("quantized"):
        display_model += " [int8]"
    if summary.get("repair_attempts", 1) > 1:
        display_model += f" [repair x{summary['repair_attempts']}]"
    eval_set = Path(summary["eval_file"]).stem
    row = (
        f"| {summary['timestamp_utc']} | `{display_model}` | {eval_set} | {summary['device']} "
        f"| {summary['n_examples']} | {summary['exact_match']:.1%} "
        f"| {summary['execution_accuracy']:.1%} |\n"
    )
    if not md_path.exists():
        md_path.write_text("# Baseline results\n\n" + header + row, encoding="utf-8")
    else:
        with md_path.open("a", encoding="utf-8") as fh:
            fh.write(row)


if __name__ == "__main__":
    raise SystemExit(main())
