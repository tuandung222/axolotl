#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
HF_HUB = Path(os.environ.get("HF_HOME", "/home/dungvpt/hf_home")) / "hub"
DEFAULT_OUTPUT = ROOT / "local_data" / "bonsai_non_reasoning_unified_v2.jsonl"
REASONING_PATTERN = re.compile(
    r"<think>|</think>|(^|\n)\s*(?:\*\*)?(?:thought|suy nghĩ)\s*:",
    re.IGNORECASE,
)
# Han/Cyrillic script intrusion into assistant output -- see investigation into
# checkpoint-1109's Vietnamese generation quality: merge_2_instruction_following
# and nemotron_v2_interactive_agent both contain assistant-authored rows with
# CJK/Cyrillic characters spliced mid-word into otherwise Latin-script text.
SCRIPT_CONTAMINATION_PATTERN = re.compile(r"[一-鿿Ѐ-ӿ]")


@dataclass(frozen=True)
class Source:
    name: str
    paths: tuple[str, ...]
    messages_field: str = "messages"
    tools_field: str | None = None
    repeat: int = 1
    drop_visible_reasoning: bool = True


@dataclass(frozen=True)
class Task:
    source: Source
    path: str
    row_group: int
    part_path: str


def latest_snapshot(repo_id: str) -> Path:
    org, name = repo_id.split("/", 1)
    root = HF_HUB / f"datasets--{org}--{name}" / "snapshots"
    snapshots = [path for path in root.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshot for {repo_id}")
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def parquet_paths(pattern: str) -> tuple[str, ...]:
    paths = tuple(str(path) for path in sorted(Path().glob(pattern)))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def snapshot_glob(repo_id: str, pattern: str) -> tuple[str, ...]:
    paths = tuple(str(path) for path in sorted(latest_snapshot(repo_id).glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"{repo_id}: {pattern}")
    return paths


def build_sources() -> list[Source]:
    # v2: dropped relative to the checkpoint-1109 mix --
    #   if_calendar_pass            -- 99.9% of rows contain two concatenated tool-result
    #                                   JSON blobs, 85% of those internally contradictory
    #   structured_outputs_pass     -- 2.53% assistant rows carry Han/Cyrillic script intrusion
    #   nvidia_if_non_thinking      -- 0.17% hedge-language rate across 250k rows, primary
    #                                   driver of the confident-fabrication/hallucination pattern
    #   merge_2_instruction_following -- 2.40% assistant rows carry Han/Cyrillic intrusion
    #                                   (10,330 rows) and 0.38% hedge rate; largest single
    #                                   contributor to both failure modes
    planning = "custom-org/planning-sft-dataset-v07022026"
    return [
        Source("planning_1_v2", snapshot_glob(planning, "1.v2/*.parquet"), repeat=2),
        Source("planning_2_v01", snapshot_glob(planning, "2.v01/*.parquet"), repeat=2),
        Source("planning_3_bucket_b", snapshot_glob(planning, "3.bucket_b/*.parquet"), repeat=2),
        Source(
            "vi_if_pass",
            snapshot_glob("tuandunghcmut/vi-if-dataset", "data/pass-*.parquet"),
        ),
        Source(
            "if_free_form_pass",
            snapshot_glob(
                "tuandunghcmut/nemotron-rl-completed-if_free_form_formatting",
                "data/pass-*.parquet",
            ),
            messages_field="completed_messages",
        ),
        Source(
            "vi_translation_pass",
            snapshot_glob("tuandunghcmut/vi-translation-task", "data/pass-*.parquet"),
        ),
        Source(
            "nemotron_v2_tool_calling",
            (str(ROOT / "local_data" / "nemotron_v2_tool_calling.parquet"),),
            tools_field="tools",
            drop_visible_reasoning=True,
        ),
        Source(
            "nemotron_v2_interactive_agent",
            (str(ROOT / "local_data" / "nemotron_v2_interactive_agent.parquet"),),
            tools_field="tools",
            drop_visible_reasoning=True,
        ),
    ]


def parse_json(value: Any) -> Any:
    if value is None or value == "":
        return None
    return json.loads(value) if isinstance(value, str) else value


def clean_messages(messages: Any, drop_visible_reasoning: bool) -> tuple[list[dict], str | None]:
    try:
        messages = parse_json(messages)
    except (TypeError, json.JSONDecodeError):
        return [], "invalid_messages_json"
    if not isinstance(messages, list) or not messages:
        return [], "invalid_messages_type"

    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            return [], "invalid_message"
        message = dict(message)
        message.pop("reasoning_content", None)
        content = message.get("content", "")
        if drop_visible_reasoning and REASONING_PATTERN.search(str(content)):
            return [], "visible_reasoning"
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            return [], "invalid_role"
        if message.get("role") == "assistant" and SCRIPT_CONTAMINATION_PATTERN.search(str(content)):
            return [], "script_contamination"
        cleaned.append(message)

    if cleaned[0].get("role") not in {"system", "user"}:
        return [], "invalid_start_role"
    if cleaned[-1].get("role") != "assistant":
        return [], "invalid_end_role"
    calls = sum(len(message.get("tool_calls") or []) for message in cleaned)
    responses = sum(message.get("role") == "tool" for message in cleaned)
    if calls != responses:
        return [], "tool_linkage_mismatch"
    return cleaned, None


def clean_tools(tools: Any) -> tuple[list[dict] | None, str | None]:
    try:
        tools = parse_json(tools)
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_tools_json"
    if tools in (None, []):
        return None, None
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        return None, "invalid_tools_type"
    return tools, None


def process_task(task: Task) -> dict[str, Any]:
    columns = [task.source.messages_field]
    if task.source.tools_field:
        columns.append(task.source.tools_field)
    table = pq.ParquetFile(task.path).read_row_group(task.row_group, columns=columns)
    counters: Counter[str] = Counter()
    Path(task.part_path).parent.mkdir(parents=True, exist_ok=True)
    with open(task.part_path, "w", encoding="utf-8") as output:
        for row in table.to_pylist():
            messages, error = clean_messages(
                row.get(task.source.messages_field), task.source.drop_visible_reasoning
            )
            if error:
                counters[f"dropped_{error}"] += 1
                continue
            tools = None
            if task.source.tools_field:
                tools, error = clean_tools(row.get(task.source.tools_field))
                if error:
                    counters[f"dropped_{error}"] += 1
                    continue
            record = {
                "messages": json.dumps(messages, ensure_ascii=False),
                "source_dataset": task.source.name,
                "tools": json.dumps(tools or [], ensure_ascii=False),
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counters["kept"] += 1
    return {
        "source": task.source.name,
        "part_path": task.part_path,
        "repeat": task.source.repeat,
        "counts": dict(counters),
    }


def create_tasks(
    sources: list[Source], parts_dir: Path, max_tasks: int | None, smoke: bool
) -> list[Task]:
    tasks = []
    for source_idx, source in enumerate(sources):
        for path_idx, path in enumerate(source.paths):
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                part = parts_dir / f"{source_idx:02d}-{path_idx:03d}-{row_group:05d}.jsonl"
                tasks.append(Task(source, path, row_group, str(part)))
                if smoke:
                    break
                if max_tasks and len(tasks) >= max_tasks:
                    return tasks
            if smoke:
                break
    return tasks


def merge_parts(results: list[dict[str, Any]], output_path: Path) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(
        os.path.getsize(result["part_path"]) * result["repeat"] for result in results
    )
    with temporary.open("wb") as output, tqdm(
        total=total, unit="B", unit_scale=True, desc="Merging JSONL", dynamic_ncols=True
    ) as progress:
        for result in sorted(results, key=lambda item: item["part_path"]):
            for _ in range(result["repeat"]):
                with open(result["part_path"], "rb") as part:
                    while chunk := part.read(16 * 1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))
    temporary.replace(output_path)


def audit_output(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    with path.open(encoding="utf-8") as dataset:
        for line in dataset:
            counts["rows"] += 1
            record = json.loads(line)
            messages = json.loads(record["messages"])
            sources[record["source_dataset"]] += 1
            if any(message.get("reasoning_content") for message in messages):
                counts["reasoning_field"] += 1
            if any(REASONING_PATTERN.search(str(message.get("content", ""))) for message in messages):
                counts["visible_reasoning"] += 1
            if "tools" in record and not isinstance(json.loads(record["tools"]), list):
                counts["invalid_tools"] += 1
    if any(counts[key] for key in ("reasoning_field", "visible_reasoning", "invalid_tools")):
        raise RuntimeError(f"Output audit failed: {dict(counts)}")
    return {"counts": dict(counts), "sources": dict(sources)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--full-audit", action="store_true")
    args = parser.parse_args()

    sources = build_sources()
    parts_dir = args.output.with_suffix(".parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    tasks = create_tasks(sources, parts_dir, args.max_tasks, args.smoke)
    print(f"sources={len(sources)} tasks={len(tasks)} workers={args.workers}", flush=True)

    results = []
    counts: dict[str, Counter[str]] = {source.name: Counter() for source in sources}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            counts[result["source"]].update(result["counts"])
            if completed % 25 == 0 or completed == len(tasks):
                print(f"completed={completed}/{len(tasks)}", flush=True)

    merge_parts(results, args.output)
    source_counts = {
        source.name: counts[source.name]["kept"] * source.repeat for source in sources
    }
    audit = {
        "counts": {"rows": sum(source_counts.values())},
        "sources": source_counts,
        "validation": "all rows validated in parallel before serialization",
    }
    if args.full_audit:
        audit = audit_output(args.output)
    manifest = {
        "output": str(args.output),
        "sources": [asdict(source) for source in sources],
        "pre_repeat_counts": {name: dict(value) for name, value in counts.items()},
        "audit": audit,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.keep_parts:
        shutil.rmtree(parts_dir)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
