#!/usr/bin/env python3
import os
import time
from pathlib import Path

from tqdm import tqdm


PID = 4119017
DATASET = Path(
    "artifacts/bonsai_internal_qat_non_reasoning/data/train.jsonl"
).resolve()


def audit_position() -> int | None:
    fd_root = Path(f"/proc/{PID}/fd")
    if not fd_root.exists():
        return None
    for fd in fd_root.iterdir():
        try:
            if fd.resolve() != DATASET:
                continue
            for line in Path(f"/proc/{PID}/fdinfo/{fd.name}").read_text().splitlines():
                if line.startswith("pos:"):
                    return int(line.split()[1])
        except (FileNotFoundError, PermissionError):
            continue
    return None


def main() -> None:
    total = DATASET.stat().st_size
    position = audit_position() or 0
    with tqdm(
        total=total,
        initial=min(position, total),
        unit="B",
        unit_scale=True,
        desc="Full JSONL audit",
        dynamic_ncols=True,
    ) as progress:
        while True:
            position = audit_position()
            if position is None:
                progress.update(total - progress.n)
                break
            progress.update(max(0, min(position, total) - progress.n))
            time.sleep(1)
    print("Audit process finished; main pipeline is moving to the next stage.")


if __name__ == "__main__":
    main()
