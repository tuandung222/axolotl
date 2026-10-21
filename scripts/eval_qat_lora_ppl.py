#!/usr/bin/env python
"""Evaluate QAT + LoRA checkpoints by answer-only perplexity.

This script loads a Transformers causal LM, patches the target linears with
KH BitLinear fake quant, then scores the same longest benchmark
conversations against each checkpoint's LoRA weights. ``--lora-mode`` selects
how those weights are loaded: ``external_peft`` attaches/switches PEFT
adapters (`adapter_config.json` checkpoints); ``internal_qat`` reloads the
BitLinear-owned ``lora_A``/``lora_B`` sidecar
(`checkpoint-*/internal_qat_lora/`) into an already use_lora=True-patched
model between checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import heapq
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axolotl.integrations.bs_qat.plugin import _replace_target_linears  # noqa: E402
from axolotl.integrations.bs_qat.bitnet_qat.internal_lora import (  # noqa: E402
    ADAPTER_WEIGHTS as INTERNAL_ADAPTER_WEIGHTS,
    load_full_checkpoint_state_dict,
    load_internal_lora,
)


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ANSWER_KEYS = (
    "deepseek_groundtruth",
    "deepseek_ground_truth",
    "deepseek_groundtrue",
    "groundtruth",
    "ground_truth",
    "reference",
    "answer",
    "output",
    "v2_output_json",
)
PLANNER_SYSTEM_PROMPT = None


@dataclass
class Example:
    sort_len: int
    instance_id: str
    prompt_text: str
    full_text: str
    prompt_len: int
    input_ids: list[int]
    labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="prism-ml/Ternary-Bonsai-4B-unpacked")
    parser.add_argument(
        "--lora-mode",
        choices=("external_peft", "internal_qat", "full_weight"),
        default="external_peft",
        help="external_peft: PEFT adapter checkpoints (adapter_config.json). "
        "internal_qat: BitLinear-owned lora_A/lora_B sidecars "
        "(checkpoint-*/internal_qat_lora/). full_weight: full (non-adapter) "
        "Trainer checkpoints (model*.safetensors), no LoRA involved.",
    )
    parser.add_argument("--checkpoint-dir", default="runs/axolotl-bs-lora-q2_0-fa3")
    parser.add_argument("--checkpoint-glob", default="checkpoint-*")
    parser.add_argument("--include-final-adapter", action="store_true")
    parser.add_argument("--include-base", action="store_true", help="Also score the fake-quantized base model without any LoRA adapter.")
    parser.add_argument("--checkpoint-limit", type=int)
    parser.add_argument("--dataset", default="custom-org/planning-benchmark-v2")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-dataset")
    parser.add_argument("--cache-dir", default="/home/dungvpt/low_bit_research/hf_home/datasets")
    parser.add_argument("--output-dir", default="runs/q2_0_checkpoint_ppl_eval")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qat-mode", default="ternary")
    parser.add_argument("--qat-group-size", type=int, default=128)
    parser.add_argument("--qat-act-quant", action="store_true")
    parser.add_argument("--skip-qat-patch", action="store_true", help="Score the loaded HF weights as-is, without wrapping linears in BitLinear.")
    parser.add_argument("--lora-r", type=int, default=128, help="internal_qat only: rank BitLinear is patched with.")
    parser.add_argument("--lora-alpha", type=int, default=256, help="internal_qat only: alpha BitLinear is patched with.")
    parser.add_argument("--target-names", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--skip-names", nargs="*", default=["lm_head"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only prepare and report the selected examples.")
    return parser.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_any_dataset(args: argparse.Namespace):
    if args.local_dataset:
        path = Path(args.local_dataset)
        if path.is_dir() and (path / "dataset_info.json").exists():
            return load_from_disk(str(path))
        data_files = str(path)
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=data_files, split=args.split)
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=data_files, split=args.split)
        if suffix == ".csv":
            return load_dataset("csv", data_files=data_files, split=args.split)
        raise ValueError(f"Unsupported --local-dataset format: {path}")

    kwargs: dict[str, Any] = {"split": args.split, "cache_dir": args.cache_dir}
    if args.dataset_config:
        return load_dataset(args.dataset, args.dataset_config, **kwargs)
    return load_dataset(args.dataset, **kwargs)


def benchmark_planner_user(row: dict[str, Any]) -> str | None:
    if not all(key in row for key in ("query", "agents_json", "groundtruth")):
        return None
    agents = json.loads(row["agents_json"]) if isinstance(row["agents_json"], str) else row["agents_json"]
    names = ", ".join(f"`{a['name']}`" for a in agents)
    blocks = "\n".join(
        f'<agent name="{a["name"]}">\n{a["description"]}\n</agent>' for a in agents
    )
    roster_text = (
        f"## Agent Roster\n"
        f"The following {len(agents)} agent(s) are available for this plan: {names}.\n"
        f"These are the ONLY valid values for any task's `agent` field.\n\n"
        f"## Available Agents\n{blocks}\n"
    )
    return (
        f"{roster_text}\n"
        f"## User Query\n{row['query']}\n\n"
        f"## Output\n"
        f"Return ONLY a JSON object of the form {{\"tasks\": [ ... ]}} that follows the policy "
        f"above. Use `task-1`, `task-2`, ... for ids. No prose, no markdown."
    )


def normalize_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    benchmark_user = benchmark_planner_user(row)
    if benchmark_user is not None:
        answer = row["groundtruth"]
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        system = PLANNER_SYSTEM_PROMPT or ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": benchmark_user})
        messages.append({"role": "assistant", "content": answer})
        return messages, answer

    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None

    if isinstance(messages, list) and messages:
        cleaned: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                cleaned.append({"role": str(role), "content": str(content)})
        for msg in reversed(cleaned):
            if msg["role"] == "assistant":
                return cleaned, msg["content"]

    answer = None
    for key in ANSWER_KEYS:
        value = row.get(key)
        if value:
            answer = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            break
    if answer is None:
        raise ValueError(f"Could not find answer field. Available keys: {sorted(row.keys())}")

    system = row.get("system_message") or row.get("system") or ""
    user = row.get("user_message") or row.get("user") or row.get("query") or row.get("prompt") or ""
    if not user and row.get("original_query"):
        user = row["original_query"]

    rebuilt = []
    if system:
        rebuilt.append({"role": "system", "content": str(system)})
    rebuilt.append({"role": "user", "content": str(user)})
    rebuilt.append({"role": "assistant", "content": str(answer)})
    return rebuilt, str(answer)


def render_example(tokenizer, row: dict[str, Any], max_length: int) -> Example | None:
    messages, answer = normalize_messages(row)
    if not answer.strip():
        return None

    prompt_messages = messages[:-1]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompt_text = f"{system}\n\nUser: {user}\nAssistant:"
        full_text = f"{prompt_text} {answer}"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if len(full_ids) > max_length:
        drop = len(full_ids) - max_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, len(prompt_ids) - drop)
    else:
        prompt_len = len(prompt_ids)

    labels = list(full_ids)
    for idx in range(min(prompt_len, len(labels))):
        labels[idx] = -100
    if not any(label != -100 for label in labels):
        return None

    instance_id = row.get("instance_id") or row.get("id") or row.get("uuid") or ""
    return Example(
        sort_len=len(full_ids),
        instance_id=str(instance_id),
        prompt_text=prompt_text,
        full_text=full_text,
        prompt_len=prompt_len,
        input_ids=list(full_ids),
        labels=labels,
    )


def select_longest_examples(tokenizer, dataset, num_samples: int, max_length: int) -> list[Example]:
    heap: list[tuple[int, int, Example]] = []
    for idx, row in enumerate(dataset):
        ex = render_example(tokenizer, dict(row), max_length)
        if ex is None:
            continue
        item = (ex.sort_len, idx, ex)
        if len(heap) < num_samples:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]


def _internal_adapter_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "internal_qat_lora"


def _has_full_model_shards(path: Path) -> bool:
    return (path / "model.safetensors").exists() or any(path.glob("model-*.safetensors"))


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    root = Path(args.checkpoint_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root

    if args.lora_mode == "internal_qat":
        # internal_qat checkpoints are full Trainer saves; the lightweight
        # lora_A/lora_B sidecar lives in a subdirectory, written by
        # InternalQATLoRASaveCallback / post_train (see plugin.py).
        checkpoints = [
            p
            for p in root.glob(args.checkpoint_glob)
            if (_internal_adapter_dir(p) / INTERNAL_ADAPTER_WEIGHTS).exists()
        ]
        checkpoints = sorted(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]) if "-" in p.name else p.name)
        if args.include_final_adapter and (_internal_adapter_dir(root) / INTERNAL_ADAPTER_WEIGHTS).exists():
            checkpoints.append(root)
    elif args.lora_mode == "full_weight":
        # full_weight has no adapter sidecar at all -- the checkpoint dir itself
        # is a full Trainer save (model.safetensors / model-*.safetensors).
        checkpoints = [p for p in root.glob(args.checkpoint_glob) if _has_full_model_shards(p)]
        checkpoints = sorted(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]) if "-" in p.name else p.name)
        if args.include_final_adapter and _has_full_model_shards(root):
            checkpoints.append(root)
    else:
        checkpoints = [p for p in root.glob(args.checkpoint_glob) if (p / "adapter_config.json").exists()]
        checkpoints = sorted(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]) if "-" in p.name else p.name)
        if args.include_final_adapter and (root / "adapter_config.json").exists():
            checkpoints.append(root)

    if args.checkpoint_limit:
        checkpoints = checkpoints[: args.checkpoint_limit]
    if not checkpoints and not args.include_base:
        raise FileNotFoundError(f"No LoRA checkpoints found under {root} with glob {args.checkpoint_glob!r}")
    return checkpoints


def pad_batch(examples: list[Example], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids = []
    labels = []
    attention_mask = []
    for ex in examples:
        pad = max_len - len(ex.input_ids)
        input_ids.append([pad_token_id] * pad + ex.input_ids)
        labels.append([-100] * pad + ex.labels)
        attention_mask.append([0] * pad + [1] * len(ex.input_ids))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


@torch.no_grad()
def score_adapter(model, examples: list[Example], micro_batch_size: int, pad_token_id: int, device: torch.device) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(examples), micro_batch_size):
        batch = examples[start : start + micro_batch_size]
        input_ids, labels, attention_mask = pad_batch(batch, pad_token_id, device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        valid = shifted_labels.ne(-100)
        if valid.any():
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_tokens += int(valid.sum().item())
        del outputs, logits, input_ids, labels, attention_mask
        torch.cuda.empty_cache()
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
        "tokens": float(total_tokens),
    }


def write_selected_examples(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "instance_id": ex.instance_id,
                "tokens": ex.sort_len,
                "prompt_tokens": ex.prompt_len,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    global PLANNER_SYSTEM_PROMPT
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_any_dataset(args)
    benchmark_root = Path(args.local_dataset).resolve().parents[1] if args.local_dataset else None
    if benchmark_root and (benchmark_root / "planeval" / "prompts.py").exists():
        sys.path.insert(0, str(benchmark_root))
        try:
            from planeval.prompts import PLANNER_SYSTEM_PROMPT as benchmark_system_prompt

            PLANNER_SYSTEM_PROMPT = benchmark_system_prompt
        except Exception as exc:
            print(f"WARNING: could not import benchmark system prompt: {exc}", file=sys.stderr)
    examples = select_longest_examples(tokenizer, dataset, args.num_samples, args.max_length)
    if len(examples) != args.num_samples:
        print(f"WARNING: selected only {len(examples)} usable examples", file=sys.stderr)
    if not examples:
        raise RuntimeError("No usable examples selected")
    write_selected_examples(out_dir / "selected_examples.jsonl", examples)
    print(
        f"Selected {len(examples)} longest examples: "
        f"max_tokens={max(e.sort_len for e in examples)} min_tokens={min(e.sort_len for e in examples)}"
    )
    if args.dry_run:
        print(f"Dry run complete. Metadata written to {out_dir / 'selected_examples.jsonl'}")
        return

    device = torch.device(args.device)
    checkpoints = discover_checkpoints(args)
    print("Checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype_from_arg(args.dtype),
        device_map={"": device.index or 0} if device.type == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False

    is_internal = args.lora_mode == "internal_qat"
    is_full_weight = args.lora_mode == "full_weight"

    if not is_internal and not is_full_weight and checkpoints:
        # external_peft only: checkpoints are PEFT adapter dirs, pre-loaded up
        # front so scoring can switch between them with set_adapter(). full_weight
        # checkpoints are full model saves, loaded into the same model in-place
        # per checkpoint below instead (see the main loop).
        first_name = checkpoints[0].name.replace("-", "_")
        model = PeftModel.from_pretrained(model, str(checkpoints[0]), adapter_name=first_name, is_trainable=False)
        for ckpt in checkpoints[1:]:
            adapter_name = ckpt.name.replace("-", "_")
            model.load_adapter(str(ckpt), adapter_name=adapter_name, is_trainable=False)

    patched = 0
    if not args.skip_qat_patch:
        # internal_qat: BitLinear must own lora_A/lora_B before any checkpoint's
        # tensors can be loaded into it -- patch with use_lora=True once, then
        # load_internal_lora() below just overwrites lora_A/lora_B per checkpoint.
        bitlinear_kwargs = {"use_lora": True, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha} if is_internal else None
        patched = _replace_target_linears(
            model,
            mode=args.qat_mode,
            group_size=args.qat_group_size,
            act_quant=args.qat_act_quant,
            bitlinear_kwargs=bitlinear_kwargs,
            target_names=tuple(args.target_names) if args.target_names else None,
            skip_names=tuple(args.skip_names),
        )
    print(f"Applied KH fake-quant wrappers: patched_linears={patched}")
    model.eval()

    rows = []
    if args.include_base:
        if is_internal:
            # No PEFT adapter to disable; zero the LoRA delta in place instead so
            # QDQ(W + 0) == QDQ(W), i.e. score the fake-quantized base only.
            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, "lora_B"):
                        module.lora_B.zero_()
        elif is_full_weight:
            # full_weight has no LoRA/adapter at all -- the model is already at
            # its just-patched, not-yet-checkpoint-loaded state here, which *is*
            # the base to score. Nothing to reset.
            pass
        elif isinstance(model, PeftModel):
            model.disable_adapter_layers()
        torch.cuda.empty_cache()
        result = score_adapter(model, examples, args.micro_batch_size, tokenizer.pad_token_id, device)
        row = {
            "checkpoint": "base",
            "adapter": "base_no_lora",
            "nll": result["nll"],
            "ppl": result["ppl"],
            "answer_tokens": int(result["tokens"]),
        }
        rows.append(row)
        print(f"base_no_lora: ppl={row['ppl']:.6f} nll={row['nll']:.6f} tokens={row['answer_tokens']}")
        if not is_internal and isinstance(model, PeftModel):
            model.enable_adapter_layers()

    for ckpt in checkpoints:
        adapter_name = ckpt.name.replace("-", "_")
        if is_internal:
            load_info = load_internal_lora(model, _internal_adapter_dir(ckpt), strict=True)
            print(f"  loaded internal-QAT LoRA from {ckpt}: tensors={load_info['loaded_tensors']}")
        elif is_full_weight:
            load_info = load_full_checkpoint_state_dict(model, ckpt)
            print(f"  loaded full-weight QAT checkpoint from {ckpt}: tensors={load_info['loaded_tensors']}")
        else:
            model.set_adapter(adapter_name)
        torch.cuda.empty_cache()
        result = score_adapter(model, examples, args.micro_batch_size, tokenizer.pad_token_id, device)
        row = {
            "checkpoint": str(ckpt),
            "adapter": adapter_name,
            "nll": result["nll"],
            "ppl": result["ppl"],
            "answer_tokens": int(result["tokens"]),
        }
        rows.append(row)
        print(f"{adapter_name}: ppl={row['ppl']:.6f} nll={row['nll']:.6f} tokens={row['answer_tokens']}")
        gc.collect()

    rows = sorted(rows, key=lambda r: r["ppl"])
    csv_path = out_dir / "checkpoint_ppl.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "adapter", "nll", "ppl", "answer_tokens"])
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"BEST: {best['checkpoint']} ppl={best['ppl']:.6f}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
