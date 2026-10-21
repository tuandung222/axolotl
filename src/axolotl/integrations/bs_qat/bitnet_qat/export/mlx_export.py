"""MLX 2-bit export (scaffolding).

MLX uses affine group quantization rather than the GGUF ``Q2_0`` ternary
container, so the cleanest path is to convert the *baked* (on-grid) HF model with
``mlx_lm`` at ``q_bits=2, q_group_size=128``. MLX requires Apple Silicon, so on
Linux/CUDA this writes a manifest with reproducible instructions instead.
"""

from __future__ import annotations

import json
from pathlib import Path


def export_mlx(
    hf_model_dir: str | Path,
    out_dir: str | Path,
    bits: int = 2,
    group_size: int = 128,
) -> dict:
    """Convert a baked HF model directory to an MLX ``bits``-bit checkpoint.

    If ``mlx_lm`` is unavailable (e.g. non-Apple hardware), writes
    ``<out_dir>/MLX_EXPORT_INSTRUCTIONS.json`` and returns ``status='scaffolding'``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_lm import convert  # type: ignore
    except Exception:
        manifest = {
            "status": "scaffolding",
            "reason": "mlx_lm not available on this platform (Apple Silicon required)",
            "hf_model_dir": str(hf_model_dir),
            "bits": bits,
            "group_size": group_size,
            "reproduce": [
                "pip install mlx mlx-lm",
                (
                    "python -c \"from mlx_lm import convert; convert("
                    f"'{hf_model_dir}', mlx_path='{out_dir}', quantize=True, "
                    f"q_bits={bits}, q_group_size={group_size})\""
                ),
            ],
        }
        path = out_dir / "MLX_EXPORT_INSTRUCTIONS.json"
        path.write_text(json.dumps(manifest, indent=2))
        return manifest

    convert(
        str(hf_model_dir),
        mlx_path=str(out_dir),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )
    return {"status": "ok", "path": str(out_dir), "bits": bits, "group_size": group_size}
