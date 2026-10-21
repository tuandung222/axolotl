#!/usr/bin/env python3
"""Normalize the LFM2.5 mixed SFT corpus into a single local JSONL file.

All reasoning/thinking traces are stripped from messages. Tool datasets keep a
top-level `tools` field so Axolotl can inject tool schemas through the chat
template.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from datasets.utils.json import ujson_loads


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = Path("/home/dungvpt/hf_home/hub")
OUT_PLAIN_PATH = REPO_ROOT / "local_data" / "lfm2_5_non_reasoning_plain.jsonl"
OUT_TOOLS_PATH = REPO_ROOT / "local_data" / "lfm2_5_non_reasoning_tools.jsonl"
SKIPPED_ROWS_PATH = REPO_ROOT / "local_data" / "lfm2_5_non_reasoning_skipped_rows.jsonl"


def latest_snapshot(org: str, name: str) -> Path:
    root = HF_HOME / f"datasets--{org}--{name}" / "snapshots"
    snapshots = [path for path in root.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"no snapshots found for {org}/{name}")
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def parse_json_maybe(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    def normalize_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}
        if isinstance(arguments, dict):
            return arguments
        if arguments is None:
            return {}
        return {"value": arguments}

    if "function" in tool_call:
        fn = dict(tool_call["function"])
        fn["arguments"] = normalize_arguments(fn.get("arguments"))
        out = dict(tool_call)
        out["function"] = fn
        out.setdefault("type", "function")
        return out

    return {
        "id": tool_call.get("id"),
        "type": "function",
        "function": {
            "name": tool_call.get("name"),
            "arguments": normalize_arguments(tool_call.get("arguments")),
        },
    }


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    items = parse_json_maybe(messages) or []
    out: list[dict[str, Any]] = []

    for message in items:
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                normalize_tool_call(tool_call)
                for tool_call in message["tool_calls"]
                if tool_call
            ]

        content = message.get("content", "")
        if message.get("role") == "assistant" and tool_calls:
            content = ""

        msg = {
            "role": message.get("role"),
            "content": content,
        }

        if message.get("name"):
            msg["name"] = message["name"]
        if message.get("tool_call_id"):
            msg["tool_call_id"] = message["tool_call_id"]
        if tool_calls:
            msg["tool_calls"] = tool_calls

        out.append(msg)

    return out


def normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    items = parse_json_maybe(tools)
    if not items:
        return None

    normalized = []
    for item in items:
        tool = dict(item)
        if "function" in tool:
            fn = dict(tool["function"])
            params = fn.get("parameters")
            if params is not None and isinstance(params, str):
                try:
                    fn["parameters"] = json.loads(params)
                except json.JSONDecodeError:
                    pass
            tool["function"] = fn
        normalized.append(tool)
    return normalized


def serialize_output_row(row: dict[str, Any]) -> tuple[str, int]:
    output_row = {
        key: value
        for key, value in row.items()
        if value not in (None, [], {})
    }
    if "tools" in output_row and not isinstance(output_row["tools"], str):
        output_row["tools"] = json.dumps(output_row["tools"], ensure_ascii=False)

    serialized = json.dumps(output_row, ensure_ascii=False)
    ujson_loads(serialized)
    return serialized, len(serialized.encode("utf-8"))


def iter_parquet_rows(paths: Iterable[Path], columns: list[str]) -> Iterable[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                yield row


def planning_rows() -> Iterable[dict[str, Any]]:
    snap = latest_snapshot("custom-org", "planning-sft-dataset-v07022026")
    subsets = ["1.v2", "2.v01", "3.bucket_b"]
    parquet_paths = []
    for subset in subsets:
        parquet_paths.extend(sorted((snap / subset).glob("*.parquet")))

    for _ in range(2):
        for row in iter_parquet_rows(parquet_paths, ["messages"]):
            yield {
                "messages": normalize_messages(row["messages"]),
                "source_dataset": "planning_sft_dataset_v07022026_x2",
            }


def if_rows() -> Iterable[dict[str, Any]]:
    sources = [
        (
            latest_snapshot("tuandunghcmut", "vi-if-dataset") / "data" / "pass-00000-of-00001.parquet",
            "messages",
            "vi_if_dataset_pass",
        ),
        (
            latest_snapshot("tuandunghcmut", "nemotron-rl-completed-if_free_form_formatting")
            / "data"
            / "pass-00000-of-00001.parquet",
            "completed_messages",
            "nemotron_if_free_form_pass",
        ),
        (
            latest_snapshot("tuandunghcmut", "nemotron-rl-completed-if_calendar_v2")
            / "data"
            / "pass-00000-of-00001.parquet",
            "completed_messages",
            "nemotron_if_calendar_pass",
        ),
        (
            latest_snapshot("tuandunghcmut", "vi-translation-task") / "data" / "pass-00000-of-00001.parquet",
            "messages",
            "vi_translation_task_pass",
        ),
        (
            latest_snapshot("tuandunghcmut", "nemotron-rl-completed-structured_outputs")
            / "data"
            / "pass-00000-of-00001.parquet",
            "completed_messages",
            "nemotron_structured_outputs_pass",
        ),
    ]

    for path, field_messages, source_name in sources:
        for row in iter_parquet_rows([path], [field_messages]):
            yield {
                "messages": normalize_messages(row[field_messages]),
                "source_dataset": source_name,
            }

    nonthinking_snap = latest_snapshot(
        "tuandunghcmut",
        "nvidia_instruction_following_if_split_v3_non_thinking",
    )
    for row in iter_parquet_rows(
        sorted((nonthinking_snap / "data").glob("train-*.parquet")),
        ["messages"],
    ):
        yield {
            "messages": normalize_messages(row["messages"]),
            "source_dataset": "nvidia_if_split_v3_non_thinking",
        }

    vi_pairs_snap = latest_snapshot("tuandunghcmut", "vi-agentic-search-toolcalling-pairs")
    for row in iter_parquet_rows(
        [vi_pairs_snap / "translation_task" / "train-00000-of-00001.parquet"],
        ["messages"],
    ):
        yield {
            "messages": normalize_messages(row["messages"]),
            "source_dataset": "vi_agentic_search_toolcalling_pairs_translation_task",
        }


def tool_rows() -> Iterable[dict[str, Any]]:
    local_sources = [
        (
            REPO_ROOT / "local_data" / "nemotron_v1_interactive_agent_train.parquet",
            "nemotron_v1_interactive_agent_train",
        ),
        (
            REPO_ROOT / "local_data" / "nemotron_v2_search.parquet",
            "nemotron_v2_search",
        ),
        (
            REPO_ROOT / "local_data" / "nemotron_v2_tool_calling.parquet",
            "nemotron_v2_tool_calling",
        ),
    ]
    for path, source_name in local_sources:
        for row in iter_parquet_rows([path], ["messages", "tools"]):
            yield {
                "messages": normalize_messages(row["messages"]),
                "tools": normalize_tools(row["tools"]),
                "source_dataset": source_name,
            }

    vi_search_snap = latest_snapshot("tuandunghcmut", "vi-agentic-search-toolcalling")
    for row in iter_parquet_rows(
        [vi_search_snap / "data" / "train-00000-of-00001.parquet"],
        ["messages", "tools"],
    ):
        yield {
            "messages": normalize_messages(row["messages"]),
            "tools": normalize_tools(row["tools"]),
            "source_dataset": "vi_agentic_search_toolcalling",
        }

    doc_slide_snap = latest_snapshot("tuandunghcmut", "doc-slide-agent-sft-trajectories")
    for row in iter_parquet_rows(
        [doc_slide_snap / "conversations" / "train-00000-of-00001.parquet"],
        ["messages", "tool_schemas"],
    ):
        yield {
            "messages": normalize_messages(row["messages"]),
            "tools": normalize_tools(row["tool_schemas"]),
            "source_dataset": "doc_slide_agent_sft_trajectories_verified_pass",
        }

    v2_snap = latest_snapshot("tuandunghcmut", "Nemotron-SFT-Agentic-v2-parquet")
    for row in iter_parquet_rows(
        [v2_snap / "interactive_agent.parquet"],
        ["messages", "tools"],
    ):
        yield {
            "messages": normalize_messages(row["messages"]),
            "tools": normalize_tools(row["tools"]),
            "source_dataset": "nemotron_sft_agentic_v2_interactive_agent",
        }

    fc_snap = latest_snapshot("tuandunghcmut", "function-calling-vietnamese")
    for row in iter_parquet_rows(
        sorted((fc_snap / "data").glob("nvidia_nemotron_fc_v1-*.parquet")),
        ["messages", "tools_json"],
    ):
        yield {
            "messages": normalize_messages(row["messages"]),
            "tools": normalize_tools(row["tools_json"]),
            "source_dataset": "function_calling_vietnamese_nvidia_nemotron_fc_v1",
        }


def write_jsonl() -> None:
    OUT_PLAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plain_total = 0
    tools_total = 0
    skipped_total = 0
    skipped_by_reason: Counter[str] = Counter()
    skipped_by_source: Counter[str] = Counter()

    with (
        OUT_PLAIN_PATH.open("w", encoding="utf-8") as plain_handle,
        OUT_TOOLS_PATH.open("w", encoding="utf-8") as tools_handle,
        SKIPPED_ROWS_PATH.open("w", encoding="utf-8") as skipped_handle,
    ):
        for iterator in (planning_rows(), if_rows(), tool_rows()):
            for row in iterator:
                if not row.get("messages"):
                    continue
                try:
                    serialized, row_bytes = serialize_output_row(row)
                except Exception as exc:  # pragma: no cover - defensive filter
                    skipped_total += 1
                    reason = f"{type(exc).__name__}: {exc}"
                    source_name = row.get("source_dataset", "unknown")
                    skipped_by_reason[reason] += 1
                    skipped_by_source[source_name] += 1
                    skipped_handle.write(
                        json.dumps(
                            {
                                "source_dataset": source_name,
                                "reason": reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                if "tools" in row:
                    tools_handle.write(serialized + "\n")
                    tools_total += 1
                else:
                    plain_handle.write(serialized + "\n")
                    plain_total += 1

    print(
        f"wrote {plain_total} plain rows to {OUT_PLAIN_PATH} and "
        f"{tools_total} tool rows to {OUT_TOOLS_PATH}"
    )
    print(f"skipped {skipped_total} rows")
    if skipped_total:
        print("skipped by source:")
        for source_name, count in skipped_by_source.most_common():
            print(f"  {source_name}: {count}")
        print("skipped by reason:")
        for reason, count in skipped_by_reason.most_common():
            print(f"  {reason}: {count}")
        print(f"details written to {SKIPPED_ROWS_PATH}")


if __name__ == "__main__":
    write_jsonl()
