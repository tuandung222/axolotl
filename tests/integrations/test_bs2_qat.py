"""Unit and GPU equivalence tests for Bonsai-2 (BS2) Hadamard + PQ2_0 Ternary QAT plugin."""

from __future__ import annotations

import os
import types
import pytest
import torch
import torch.nn as nn

from axolotl.integrations.bs2_qat import (
    BS2QATArgs,
    BS2QATPlugin,
    Bonsai2LinearFakeQuant,
    fwht_fast,
    get_signs_for_width,
    infer_bonsai2_block_size,
    load_bs2_internal_lora,
    load_signs_table,
    save_bs2_internal_lora,
    snap_bs2_qat_to_standard_linear,
    ternary_quantize_pq2_0_no_grad,
)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_builtin_signs_table_exact_widths():
    table = load_signs_table("builtin")
    expected_widths = {2048, 2560, 4096, 5120, 6144, 9216, 12288, 17408}
    assert expected_widths.issubset(set(table.keys()))
    for w in expected_widths:
        s = table[w]
        assert s.shape == (w,)
        assert set(s.unique().tolist()).issubset({-1.0, 1.0})
    # Verify prefix of 2048 and 6144 matches Prism ML Ternary-Bonsai-2 hadamard.json
    assert table[2048][:5].tolist() == [-1.0, -1.0, -1.0, 1.0, 1.0]
    assert table[6144][:5].tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("block_size", [512, 1024, 4096])
def test_fwht_fast_involution_and_orthogonality(block_size: int):
    dev = _device()
    torch.manual_seed(42)
    signs = get_signs_for_width(block_size, seed=7, device=dev)
    x = torch.randn(8, block_size, device=dev, dtype=torch.float32, requires_grad=True)

    y = fwht_fast(x, block_size, signs, inverse=False)
    x_rec = fwht_fast(y, block_size, signs, inverse=True)

    # 1. Exact involution T^-1(T(x)) == x
    assert torch.allclose(x_rec, x, atol=1e-5, rtol=1e-5)

    # 2. Orthogonal norm preservation ||T(x)||_2 == ||x||_2
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)

    # 3. Autograd backward check
    grad_out = torch.randn_like(y)
    (grad_x,) = torch.autograd.grad(y, x, grad_out)
    expected_grad_x = fwht_fast(grad_out, block_size, signs, inverse=True)
    assert torch.allclose(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)


def test_pq2_0_least_squares_exact_trit_recovery():
    dev = _device()
    torch.manual_seed(123)
    rows, width, group_size = 64, 2048, 128
    num_groups = width // group_size

    # Generate synthetic true trits in {-1, 0, +1} with ~1/3 probability each and FP16 scales
    true_trits = torch.randint(-1, 2, (rows, num_groups, group_size), device=dev, dtype=torch.int8)
    # Ensure every group has at least one non-zero element, except row 0 which is all-zero (like in_proj_qkv)
    true_trits[:, :, 0] = 1
    true_trits[0, :, :] = 0
    true_scales = (torch.rand(rows, num_groups, device=dev, dtype=torch.float32) * 0.05 + 0.005).to(
        torch.float16
    )
    true_scales[0, :] = 0.0
    w_rot = (true_trits.to(torch.float32) * true_scales.to(torch.float32).unsqueeze(-1)).reshape(
        rows, width
    )

    w_hat, q_hard, scales_fp16 = ternary_quantize_pq2_0_no_grad(w_rot, group_size=group_size)
    assert not w_hat.isnan().any()
    assert torch.equal(q_hard, true_trits)
    assert torch.allclose(scales_fp16.float(), true_scales.float(), atol=1e-4, rtol=1e-3)
    assert torch.allclose(w_hat, w_rot, atol=1e-5)


def test_linear_fake_quant_modes_and_gradients():
    dev = _device()
    torch.manual_seed(99)
    in_features, out_features, block_size = 512, 256, 512
    signs = get_signs_for_width(in_features, seed=11, device=dev)

    base_linear = nn.Linear(in_features, out_features, bias=True, device=dev, dtype=torch.float32)
    layer_unfolded = Bonsai2LinearFakeQuant.from_linear(
        base_linear, block_size=block_size, signs=signs, store_rotated=False
    )
    layer_rotated = Bonsai2LinearFakeQuant.from_linear(
        base_linear, block_size=block_size, signs=signs, store_rotated=True
    )

    x1 = torch.randn(4, 16, in_features, device=dev, dtype=torch.float32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    grad_y = torch.randn(4, 16, out_features, device=dev, dtype=torch.float32)

    y1 = layer_unfolded(x1)
    y2 = layer_rotated(x2)
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)

    y1.backward(grad_y)
    y2.backward(grad_y)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4, rtol=1e-4)
    grad_w1_rot = fwht_fast(layer_unfolded.weight.grad, block_size, signs, inverse=False)
    assert torch.allclose(grad_w1_rot, layer_rotated.weight.grad, atol=1e-4, rtol=1e-4)


def test_internal_qat_lora_and_sidecar_roundtrip(tmp_path):
    dev = _device()
    torch.manual_seed(77)
    in_features, out_features, block_size = 512, 256, 512
    signs = get_signs_for_width(in_features, seed=3, device=dev)

    base_linear = nn.Linear(in_features, out_features, bias=False, device=dev, dtype=torch.float32)
    q_no_lora = Bonsai2LinearFakeQuant.from_linear(
        base_linear, block_size=block_size, signs=signs, use_lora=False
    )
    q_lora = Bonsai2LinearFakeQuant.from_linear(
        base_linear,
        block_size=block_size,
        signs=signs,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
    )

    x = torch.randn(2, 8, in_features, device=dev, dtype=torch.float32)
    # Step 0: lora_B is initialized to 0 so forward output is identical to base QAT layer
    assert torch.allclose(q_no_lora(x), q_lora(x), atol=1e-5)

    # Step 0 backward: base weight is frozen, lora_B receives non-zero grad immediately
    loss0 = q_lora(x).pow(2).mean()
    loss0.backward()
    assert not q_lora.weight.requires_grad
    assert q_lora.weight.grad is None
    assert q_lora.lora_B.grad is not None and q_lora.lora_B.grad.abs().sum() > 0

    # Take 1 gradient step on lora_B -> Step 1 backward gives non-zero grad on BOTH lora_A and lora_B
    with torch.no_grad():
        q_lora.lora_B.sub_(0.1 * q_lora.lora_B.grad)
    q_lora.zero_grad()
    loss1 = q_lora(x).pow(2).mean()
    loss1.backward()
    assert q_lora.lora_A.grad is not None and q_lora.lora_A.grad.abs().sum() > 0
    assert q_lora.lora_B.grad is not None and q_lora.lora_B.grad.abs().sum() > 0

    y_before_save = q_lora(x)
    container = nn.ModuleDict({"proj": q_lora})
    info = save_bs2_internal_lora(container, tmp_path / "lora_ckpt")
    assert info["tensors"] == 2

    with torch.no_grad():
        q_lora.lora_B.zero_()
    assert not torch.allclose(q_lora(x), y_before_save)

    loaded = load_bs2_internal_lora(container, tmp_path / "lora_ckpt")
    assert loaded == 2
    assert torch.allclose(q_lora(x), y_before_save, atol=1e-6)

    # Snap to standard nn.Linear and verify output matches 100%
    snapped_count = snap_bs2_qat_to_standard_linear(container)
    assert snapped_count == 1
    assert isinstance(container["proj"], nn.Linear)
    assert torch.allclose(container["proj"](x), y_before_save, atol=1e-5)


class _DummyBlock(nn.Module):
    def __init__(self, hidden_size: int = 512, inter_size: int = 1024, dev: torch.device | None = None):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
                "k_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
                "v_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
                "o_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
            }
        )
        self.linear_attn = nn.ModuleDict(
            {
                "in_proj_qkv": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
                "in_proj_z": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
                "in_proj_a": nn.Linear(hidden_size, 16, bias=False, device=dev),
                "in_proj_b": nn.Linear(hidden_size, 16, bias=False, device=dev),
                "out_proj": nn.Linear(hidden_size, hidden_size, bias=False, device=dev),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden_size, inter_size, bias=False, device=dev),
                "up_proj": nn.Linear(hidden_size, inter_size, bias=False, device=dev),
                "down_proj": nn.Linear(inter_size, hidden_size, bias=False, device=dev),
            }
        )


class _DummyQwen35VLM(nn.Module):
    def __init__(self, dev: torch.device):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=2048)
        )
        self.model = nn.ModuleDict(
            {
                "visual": nn.ModuleDict({"patch_proj": nn.Linear(512, 512, bias=False, device=dev)}),
                "language_model": nn.ModuleDict(
                    {
                        "layers": nn.ModuleList([_DummyBlock(512, 1024, dev=dev)]),
                        "lm_head": nn.Linear(512, 1024, bias=False, device=dev),
                    }
                ),
            }
        )


def test_bs2_qat_plugin_patching_and_skip_rules():
    dev = _device()
    model = _DummyQwen35VLM(dev)
    assert infer_bonsai2_block_size(model) == 512

    plugin = BS2QATPlugin()
    cfg = BS2QATArgs(
        bs2_qat=True,
        bs2_qat_block_size=512,
        bs2_qat_lora_mode="internal_qat",
        bs2_qat_lora_r=8,
        bs2_qat_lora_alpha=16.0,
    )
    plugin.post_model_load(cfg, model)

    blk = model.model["language_model"]["layers"][0]
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(blk.self_attn[name], Bonsai2LinearFakeQuant)
    for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
        assert isinstance(blk.linear_attn[name], Bonsai2LinearFakeQuant)
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert isinstance(blk.mlp[name], Bonsai2LinearFakeQuant)

    # Skipped projections & Vision Tower remain untouched nn.Linear
    assert type(blk.linear_attn["in_proj_a"]) is nn.Linear
    assert type(blk.linear_attn["in_proj_b"]) is nn.Linear
    assert type(model.model["language_model"]["lm_head"]) is nn.Linear
    assert type(model.model["visual"]["patch_proj"]) is nn.Linear

    # Validation check: internal_qat + external adapter must raise ValueError
    bad_cfg = BS2QATArgs(bs2_qat=True, bs2_qat_lora_mode="internal_qat")
    bad_cfg_with_adapter = types.SimpleNamespace(**bad_cfg.model_dump(), adapter="lora")
    try:
        plugin._validate_cfg(bad_cfg_with_adapter)
        assert False, "Expected ValueError when adapter='lora' with internal_qat"
    except ValueError:
        pass


def test_frozen_weight_cache_and_bf16_ste_grid_parity():
    dev = _device()
    torch.manual_seed(2026)
    in_features, out_features, block_size = 512, 256, 512
    signs = get_signs_for_width(in_features, seed=19, device=dev)

    base_linear = nn.Linear(in_features, out_features, bias=False, device=dev, dtype=torch.bfloat16)
    layer = Bonsai2LinearFakeQuant.from_linear(
        base_linear, block_size=block_size, signs=signs, use_lora=False
    )
    # 1. Verify _STEGate in BF16 matches quantized inference weight 100.0000% (no BF16 w + (wq - w) drift)
    x = torch.randn(2, 8, in_features, device=dev, dtype=torch.bfloat16)
    y_train = layer(x)
    snapped_linear = layer.to_standard_linear()
    y_infer = snapped_linear(x)
    assert torch.equal(y_train, y_infer)

    # 2. Verify frozen-weight cache populates when weight.requires_grad is False and invalidates on clear/mutate
    layer.weight.requires_grad_(False)
    assert layer._cached_weight is None
    y_cached_1 = layer(x)
    assert layer._cached_weight is not None
    cached_ref = layer._cached_weight
    y_cached_2 = layer(x)
    assert layer._cached_weight is cached_ref
    assert torch.equal(y_cached_1, y_cached_2)

    # Mutating weight in-place bumps _version and invalidates cache automatically
    with torch.no_grad():
        layer.weight.mul_(-1.0)
    y_mutated = layer(x)
    assert layer._cached_weight is not cached_ref
    assert torch.allclose(y_mutated, -y_cached_1, atol=1e-3, rtol=1e-3)

    layer.clear_weight_cache()
    assert layer._cached_weight is None

