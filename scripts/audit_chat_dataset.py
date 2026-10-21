#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def audit_jsonl(path: Path, limit: int) -> None:
    role_counts = Counter()
    rows = 0
    bad_rows = 0
    char_lens = []
    assistant_turns = []
    tool_turns = []
    duplicate_keys = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                bad_rows += 1
                if rows >= limit:
                    break
                continue

            digest = hashlib.sha1(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            duplicate_keys[digest] += 1

            total_chars = 0
            local_assistant = 0
            local_tool = 0
            for msg in messages:
                role = msg.get("role")
                role_counts[role] += 1
                if role == "assistant":
                    local_assistant += 1
                elif role == "tool":
                    local_tool += 1
                content = msg.get("content")
                if isinstance(content, str):
                    total_chars += len(content)

            char_lens.append(total_chars)
            assistant_turns.append(local_assistant)
            tool_turns.append(local_tool)
            if rows >= limit:
                break

    dup_rows = sum(v - 1 for v in duplicate_keys.values() if v > 1)
    print(f"RAW {path}")
    print(
        {
            "sampled_rows": rows,
            "bad_rows": bad_rows,
            "duplicate_rows": dup_rows,
            "role_counts": dict(role_counts),
            "chars_p50": int(quantile(char_lens, 0.5)),
            "chars_p90": int(quantile(char_lens, 0.9)),
            "chars_p99": int(quantile(char_lens, 0.99)),
            "assistant_turns_p50": int(quantile(assistant_turns, 0.5)),
            "assistant_turns_p90": int(quantile(assistant_turns, 0.9)),
            "tool_turns_p90": int(quantile(tool_turns, 0.9)),
        }
    )


def audit_arrow(prepared_path: Path, shard_glob: str, limit: int) -> None:
    arrow_path = next(prepared_path.glob(shard_glob))
    ds = load_dataset("arrow", data_files=str(arrow_path), split="train")
    seq_lens = []
    label_ratios = []
    first_label_positions = []

    for idx in range(min(limit, len(ds))):
        ex = ds[idx]
        labels = ex["labels"]
        seq_len = len(ex["input_ids"])
        non_ignored = sum(1 for x in labels if x != -100)
        first = next((i for i, x in enumerate(labels) if x != -100), seq_len)
        seq_lens.append(seq_len)
        label_ratios.append(non_ignored / seq_len if seq_len else 0.0)
        first_label_positions.append(first / seq_len if seq_len else 0.0)

    print(f"TOKENIZED {arrow_path}")
    print(
        {
            "sampled_rows": min(limit, len(ds)),
            "seq_len_p50": int(quantile(seq_lens, 0.5)),
            "seq_len_p90": int(quantile(seq_lens, 0.9)),
            "seq_len_p99": int(quantile(seq_lens, 0.99)),
            "label_ratio_p10": round(quantile(label_ratios, 0.1), 4),
            "label_ratio_p50": round(quantile(label_ratios, 0.5), 4),
            "label_ratio_p90": round(quantile(label_ratios, 0.9), 4),
            "first_label_frac_p10": round(quantile(first_label_positions, 0.1), 4),
            "first_label_frac_p50": round(quantile(first_label_positions, 0.5), 4),
            "first_label_frac_p90": round(quantile(first_label_positions, 0.9), 4),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="*", default=[])
    parser.add_argument("--prepared-path")
    parser.add_argument("--shard-glob", default="*/data-00000-of-*.arrow")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    for item in args.jsonl:
        audit_jsonl(Path(item), args.limit)

    if args.prepared_path:
        audit_arrow(Path(args.prepared_path), args.shard_glob, args.limit)


if __name__ == "__main__":
    main()
