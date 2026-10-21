"""Import the PrismML ``llama.cpp`` fork's ``gguf`` package (knows Q1_0/Q2_0).

The stock PyPI ``gguf`` package does not define the Bonsai ternary types, so we
prefer the fork's ``gguf-py`` if a checkout exists next to this repo
(``Bonsai-demo/llama.cpp/gguf-py``). Falls back to whatever ``gguf`` is installed
as long as it knows ``Q2_0``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_fork_gguf() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "llama.cpp" / "gguf-py"
        if (cand / "gguf" / "constants.py").exists():
            return cand
    return None


_fork = _find_fork_gguf()
if _fork is not None and str(_fork) not in sys.path:
    sys.path.insert(0, str(_fork))

import gguf  # noqa: E402

if not hasattr(gguf.GGMLQuantizationType, "Q2_0"):
    raise ImportError(
        "The available `gguf` package does not define Q2_0. Install the fork:\n"
        "    pip install -e llama.cpp/gguf-py\n"
        "or ensure Bonsai-demo/llama.cpp/gguf-py exists."
    )

__all__ = ["gguf"]
