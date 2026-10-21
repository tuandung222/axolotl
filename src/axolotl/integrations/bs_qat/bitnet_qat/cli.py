"""Command-line interface for the ternary QAT toolkit.

Subcommands:
  inspect   List a model's nn.Linear layers (to discover include patterns).
  train     Convert -> QAT (distill) -> bake -> save HF dir + ternary safetensors.
  export    Baked HF dir -> F16 GGUF (convert_hf_to_gguf) -> Q2_0 GGUF (+ MLX).
  ptq       Any HF model -> F16 GGUF -> Q2_0 GGUF (post-training ternary, no training).

Examples:
  python -m bitnet_qat.cli inspect --model Qwen/Qwen3-0.6B
  python -m bitnet_qat.cli train  --config bitnet_qat/configs/qwen3.yaml
  python -m bitnet_qat.cli export --config bitnet_qat/configs/qwen3.yaml --hf-dir qat/runs/qwen3-0.6b-ternary
  python -m bitnet_qat.cli ptq    --model Qwen/Qwen3-0.6B --out-dir qat/exports/qwen3-ptq
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Force the torch backend before any transformers import (avoids TF/Keras import).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

logger = logging.getLogger("bitnet_qat.cli")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT_SCRIPT = REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_convert_hf_to_gguf(hf_dir: str | Path, out_f16: str | Path) -> None:
    """Invoke the fork's convert_hf_to_gguf.py to produce an F16 GGUF."""
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"{CONVERT_SCRIPT} not found. Clone the fork:\n"
            "  git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp"
        )
    Path(out_f16).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(hf_dir),
        "--outtype",
        "f16",
        "--outfile",
        str(out_f16),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _gguf_quant_plan(
    mode: str, export_cfg: dict
) -> tuple[list[str] | None, list[str] | None, str]:
    """Decide ``(ternary_patterns, binary_patterns, file_suffix)`` for GGUF export.

    - Global ``ternary`` -> default ternary projection set (``Q2_0``).
    - Global ``binary``  -> default projection set packed as ``Q1_0``.
    - Explicit ``export.binary_patterns`` / ``export.ternary_patterns`` (GGUF
      tensor-name globs) -> mixed/custom selection.
    """
    from .export.requantize_gguf import DEFAULT_QUANT_PATTERNS

    binary_patterns = export_cfg.get("binary_patterns")
    ternary_patterns = export_cfg.get("ternary_patterns")
    if binary_patterns is None and ternary_patterns is None:
        if mode == "binary":
            return [], list(DEFAULT_QUANT_PATTERNS), "Q1_0"
        return None, None, "Q2_0"
    if ternary_patterns is None and binary_patterns:
        return [], list(binary_patterns), "Q1_0"
    if binary_patterns is None and ternary_patterns:
        return list(ternary_patterns), None, "Q2_0"
    return list(ternary_patterns), list(binary_patterns), "bonsai"


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_inspect(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    from .convert import list_linear_modules

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    rows = list_linear_modules(model)
    width = max((len(n) for n, _, _ in rows), default=4)
    print(f"{'module':<{width}}  in_features  out_features  q2_0_ok")
    for name, in_f, out_f in rows:
        ok = "yes" if in_f % args.group_size == 0 else "NO"
        print(f"{name:<{width}}  {in_f:>11}  {out_f:>12}  {ok}")
    print(f"\n{len(rows)} Linear layers.")


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import build_convert_config, load_yaml
    from .convert import convert_model
    from .export.bake import bake_bitlinear_to_linear
    from .export.safetensors_export import export_safetensors
    from .train import build_hf_dataset, build_stage_configs, load_teacher, run_training_stage

    cfg = load_yaml(args.config)
    model_name = cfg["model"]
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    convert_cfg = build_convert_config(cfg)
    train_cfg = cfg.get("train", {})

    stages = build_stage_configs(train_cfg)
    if args.stage != "both":
        stages = [(n, c) for (n, c) in stages if n == args.stage]
        if not stages:
            raise SystemExit(f"stage {args.stage!r} not found in config; "
                             f"available: {[n for n, _ in build_stage_configs(train_cfg)]}")
    if args.max_steps is not None:
        for _, c in stages:
            c.max_steps = args.max_steps

    base_save = Path(train_cfg.get("save_dir", "qat/runs/bitnet"))
    qat_ckpt = base_save / "stage_qat" / "qat_state.pt"

    dtype = torch.bfloat16 if stages[0][1].bf16 else torch.float32
    logger.info("Loading student %s", model_name)
    attn_kwargs = {}
    if train_cfg.get("flash_attention"):
        fa_val = train_cfg.get("flash_attention")
        
        # Determine if we should load from HF Kernels Hub
        kernel_id = None
        if isinstance(fa_val, str) and "/" in fa_val:
            kernel_id = fa_val
        elif str(fa_val).lower() in ["3", "flash_attention_3", "flash-attn3"]:
            kernel_id = "kernels-community/flash-attn3"
            
        if kernel_id:
            try:
                from kernels import get_kernel
                import sys
                import types
                fa_kernel = get_kernel(kernel_id, version=1)
                # Monkey-patch flash_attn so transformers can use it
                fa_module = types.ModuleType("flash_attn")
                fa_module.flash_attn_func = fa_kernel.flash_attn_func
                fa_module.flash_attn_varlen_func = fa_kernel.flash_attn_varlen_func
                sys.modules["flash_attn"] = fa_module
                attn_kwargs["attn_implementation"] = "flash_attention_2" # transformers checks this string
                logger.info(f"Successfully loaded flash attention from huggingface kernels: {kernel_id}")
            except Exception as e:
                logger.warning(f"Failed to load kernel {kernel_id}: {e}. Falling back to SDPA.")
                attn_kwargs["attn_implementation"] = "sdpa"
        else:
            try:
                import flash_attn
                attn_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                attn_kwargs["attn_implementation"] = "sdpa"
            logger.info("FlashAttention-2 not found, falling back to SDPA")

    student = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=trust_remote_code, **attn_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-pad for causal-LM training so RoPE positions and the label shift stay
    # aligned (left-pad is for batch generation); the collator masks pads to -100.
    tokenizer.padding_side = "right"

    use_lora = any(scfg.use_lora for _, scfg in stages)
    lora_kwargs = {}
    if use_lora:
        logger.info("QA-LoRA mode enabled: initializing BitLinear with LoRA matrices.")
        lora_cfg = stages[0][1]
        lora_kwargs = {
            "use_lora": True,
            "lora_r": lora_cfg.lora_r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
        }

    report = convert_model(student, convert_cfg, **lora_kwargs)
    logger.info("BitLinear conversion: %s", report.summary())

    if use_lora:
        logger.info("Freezing non-LoRA parameters to ensure only adapters are trained.")
        for name, param in student.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Resume the quantized shadow weights from a prior QAT stage when starting
    # directly at the distill stage.
    if args.stage == "distill":
        ckpt = Path(args.resume_from) if args.resume_from else qat_ckpt
        if not ckpt.exists():
            raise SystemExit(f"--stage distill needs a QAT checkpoint; {ckpt} not found "
                             f"(run `--stage qat` first).")
        logger.info("Resuming QAT shadow weights from %s", ckpt)
        student.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    teachers: dict = {}
    for name, scfg in stages:
        teacher = None
        if scfg.distill_alpha > 0.0:
            tname = scfg.teacher or model_name
            key = (tname, scfg.teacher_load_in_4bit)
            if key not in teachers:
                logger.info("Loading frozen teacher %s (4bit=%s)", tname, scfg.teacher_load_in_4bit)
                teachers[key] = load_teacher(
                    tname, dtype, scfg.teacher_load_in_4bit, trust_remote_code=trust_remote_code
                )
            teacher = teachers[key]

        if scfg.datasets:
            logger.info(
                "Building mixed dataset: %s",
                ", ".join(f"{s['name']}[{s.get('split', scfg.dataset_split)}]" for s in scfg.datasets),
            )
        else:
            logger.info("Building dataset %s[%s]", scfg.dataset, scfg.dataset_split)
        ds = build_hf_dataset(tokenizer, scfg)

        scfg.save_dir = str(base_save / f"stage_{name}")
        logger.info("===== Training stage: %s =====", name)
        stage_resume = args.resume_from_checkpoint if (args.stage in (None, "both") or name == args.stage) else None
        run_training_stage(student, teacher, ds, tokenizer, scfg, resume_from_checkpoint=stage_resume)

        if name == "qat":
            qat_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), qat_ckpt)
            logger.info("Saved QAT checkpoint -> %s", qat_ckpt)

    if args.stage == "qat":
        print(f"\nQAT stage complete. Checkpoint: {qat_ckpt}\n"
              f"Next:\n  python -m bitnet_qat.cli train --config {args.config} --stage distill")
        return

    base_save.mkdir(parents=True, exist_ok=True)
    if cfg.get("export", {}).get("safetensors", True):
        info = export_safetensors(student, base_save / "bitnet.safetensors", group_size=convert_cfg.group_size)
        logger.info("Saved safetensors: %s", info)

    n_baked = bake_bitlinear_to_linear(student)
    logger.info("Baked %d BitLinear layers to nn.Linear for export.", n_baked)
    student.save_pretrained(base_save)
    tokenizer.save_pretrained(base_save)
    logger.info("Saved baked HF model to %s", base_save)
    print(f"\nDone. Next:\n  python -m bitnet_qat.cli export --config {args.config} --hf-dir {base_save}")


def cmd_export(args: argparse.Namespace) -> None:
    from .config import load_yaml
    from .export.mlx_export import export_mlx
    from .export.requantize_gguf import requantize_gguf

    cfg = load_yaml(args.config) if args.config else {}
    export_cfg = cfg.get("export", {})
    mode = cfg.get("mode", "ternary")
    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir or export_cfg.get("out_dir", hf_dir / "gguf"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ternary_patterns, binary_patterns, suffix = _gguf_quant_plan(mode, export_cfg)
    name = hf_dir.name
    f16_path = out_dir / f"{name}-f16.gguf"
    out_path = out_dir / f"{name}-{suffix}.gguf"

    run_convert_hf_to_gguf(hf_dir, f16_path)
    stats = requantize_gguf(
        f16_path,
        out_path,
        patterns=ternary_patterns,
        binary_patterns=binary_patterns,
        quantize_embeddings=args.quantize_embeddings,
        group_size=cfg.get("group_size", 128),
    )
    logger.info("GGUF export: %s -> %s", stats, out_path)
    print(
        f"Wrote {out_path} "
        f"(Q1_0={stats['quantized_q1_0']} Q2_0={stats['quantized_q2_0']} tensors)"
    )

    if args.mlx or export_cfg.get("mlx_2bit", False):
        info = export_mlx(hf_dir, out_dir / f"{name}-mlx-2bit")
        logger.info("MLX export: %s", info)


def cmd_ptq(args: argparse.Namespace) -> None:
    """Post-training quantization: no QAT, just convert + absmax Q1_0/Q2_0 quantize."""
    from .export.requantize_gguf import requantize_gguf

    mode = "binary" if args.binary else "ternary"
    ternary_patterns, binary_patterns, suffix = _gguf_quant_plan(mode, {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.model).name
    f16_path = out_dir / f"{name}-f16.gguf"
    out_path = out_dir / f"{name}-{suffix}.gguf"

    run_convert_hf_to_gguf(args.model, f16_path)
    stats = requantize_gguf(
        f16_path,
        out_path,
        patterns=ternary_patterns,
        binary_patterns=binary_patterns,
        quantize_embeddings=args.quantize_embeddings,
        group_size=args.group_size,
    )
    logger.info("PTQ export: %s -> %s", stats, out_path)
    print(
        f"Wrote {out_path} "
        f"(Q1_0={stats['quantized_q1_0']} Q2_0={stats['quantized_q2_0']} tensors)"
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bitnet_qat", description=__doc__)
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="list Linear layers of a model")
    pi.add_argument("--model", required=True)
    pi.add_argument("--group-size", type=int, default=128)
    pi.add_argument("--trust-remote-code", action="store_true")
    pi.set_defaults(func=cmd_inspect)

    pt = sub.add_parser("train", help="two-stage QAT + distill, then bake + save")
    pt.add_argument("--config", required=True)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument(
        "--stage",
        choices=["pretrain", "qat", "distill", "both"],
        default="both",
        help="run a single stage (pretrain warm-up / qat / distill) or all of them (both, default)",
    )
    pt.add_argument(
        "--resume-from",
        default=None,
        help="QAT checkpoint (.pt) to resume shadow weights from for --stage distill",
    )
    pt.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="HF Trainer checkpoint dir to resume the selected --stage (qat or distill) from (model+optimizer+scheduler+step)",
    )
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("export", help="baked HF dir -> Q2_0 GGUF (+MLX)")
    pe.add_argument("--config", default=None)
    pe.add_argument("--hf-dir", required=True)
    pe.add_argument("--out-dir", default=None)
    pe.add_argument("--quantize-embeddings", action="store_true")
    pe.add_argument("--mlx", action="store_true")
    pe.set_defaults(func=cmd_export)

    pq = sub.add_parser("ptq", help="post-training Q1_0/Q2_0 (no training)")
    pq.add_argument("--model", required=True)
    pq.add_argument("--out-dir", required=True)
    pq.add_argument("--group-size", type=int, default=128)
    pq.add_argument("--binary", action="store_true", help="pack 1-bit Q1_0 instead of ternary Q2_0")
    pq.add_argument("--quantize-embeddings", action="store_true")
    pq.set_defaults(func=cmd_ptq)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
