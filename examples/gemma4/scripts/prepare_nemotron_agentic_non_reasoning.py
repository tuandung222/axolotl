#!/usr/bin/env python
"""Prepare Nemotron Agentic parquet data for non-reasoning Gemma 4 SFT."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET_ID = "tuandunghcmut/Nemotron-SFT-Agentic-v2-search-toolcalling-parquet"
SUBSETS = ("search.parquet", "tool_calling.parquet")
DROP_KEYS = {"reasoning_content", "reasoning", "thinking"}
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        if not value.strip():
            return default
        return json.loads(value)
    return value


def strip_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_reasoning(val) for key, val in value.items() if key not in DROP_KEYS}
    if isinstance(value, list):
        return [strip_reasoning(item) for item in value]
    if isinstance(value, str):
        return THINK_RE.sub("", value).strip()
    return value


def sanitize_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    tool = dict(tool)
    function = tool.get("function")
    if isinstance(function, dict):
        function = dict(function)
        if function.get("name") is None:
            function["name"] = "unknown"
        if function.get("description") is None:
            function["description"] = ""
        if function.get("parameters") is None:
            function["parameters"] = {"type": "object", "properties": {}, "required": []}
        tool["function"] = function
    return tool


def sanitize_message(message: dict[str, Any]) -> dict[str, Any]:
    message = dict(message)
    if message.get("content") is None:
        message["content"] = ""
    if message.get("role") == "tool" and message.get("name") is None:
        message["name"] = "unknown"
    if isinstance(message.get("tool_calls"), list):
        tool_calls = []
        for tool_call in message["tool_calls"]:
            if not isinstance(tool_call, dict):
                continue
            tool_call = dict(tool_call)
            function = tool_call.get("function")
            if isinstance(function, dict):
                function = dict(function)
                if function.get("name") is None:
                    function["name"] = "unknown"
                if function.get("arguments") is None:
                    function["arguments"] = "{}"
                tool_call["function"] = function
            tool_calls.append(tool_call)
        message["tool_calls"] = tool_calls
    return message


def normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    parsed = parse_jsonish(tools, None)
    if not parsed:
        return None
    if isinstance(parsed, list):
        return [sanitize_tool_schema(tool) for tool in strip_reasoning(parsed) if isinstance(tool, dict)]
    return None


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    parsed = parse_jsonish(messages, [])
    if not isinstance(parsed, list):
        raise ValueError("messages must be a list")
    cleaned = strip_reasoning(parsed)
    return [sanitize_message(msg) for msg in cleaned if isinstance(msg, dict) and msg.get("role")]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"dataset_id": args.dataset_id, "subsets": {}}

    for subset in SUBSETS:
        parquet_path = hf_hub_download(args.dataset_id, subset, repo_type="dataset")
        table = pq.read_table(parquet_path)
        subset_rows = []
        dropped = 0
        for idx, record in enumerate(table.to_pylist()):
            try:
                messages = normalize_messages(record.get("messages"))
                tools = normalize_tools(record.get("tools"))
            except Exception:
                dropped += 1
                continue
            if not messages:
                dropped += 1
                continue
            out = {
                "conversations": messages,
                "source_subset": subset.replace(".parquet", ""),
                "source_index": idx,
            }
            if tools:
                out["tools"] = tools
            subset_rows.append(out)
        rows.extend(subset_rows)
        summary["subsets"][subset] = {"kept": len(subset_rows), "dropped": dropped}

    random.Random(args.seed).shuffle(rows)
    eval_count = int(len(rows) * args.eval_ratio)
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    summary.update(
        {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "drop_keys": sorted(DROP_KEYS),
            "removed_think_tags": True,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
