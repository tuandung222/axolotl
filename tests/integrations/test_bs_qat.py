"""Unit coverage for the BS QAT Axolotl integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from axolotl.integrations.bs_qat.bitnet_qat import bitlinear as bitlinear_mod
from axolotl.integrations.bs_qat.bitnet_qat.bitlinear import (
    BitLinear,
    TernaryBitLinear,
)
from axolotl.integrations.bs_qat.bitnet_qat.convert import (
    ConvertConfig,
    convert_model,
)
from axolotl.integrations.bs_qat.bitnet_qat.gguf_quant import (
    pack_q2_0,
    unpack_q2_0,
)
from axolotl.integrations.bs_qat.bitnet_qat.internal_lora import (
    ADAPTER_WEIGHTS,
    load_full_checkpoint_state_dict,
    load_internal_lora,
    save_internal_lora,
)
from safetensors.torch import save_file
from axolotl.integrations.bs_qat.bitnet_qat.ternary import (
    dequantize,
    quantize_ternary,
)
from axolotl.integrations.bs_qat import plugin as plugin_mod
from axolotl.integrations.bs_qat.plugin import BSQATPlugin
from axolotl.utils.dict import DictDefault


class TinyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 64)
        self.k_proj = nn.Linear(128, 64)
        self.other = nn.Linear(128, 64)


class TinyPluginModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 64)
        self.bad_proj = nn.Linear(127, 64)
        self.lm_head = nn.Linear(128, 10)


def test_convert_model_only_converts_matching_include_patterns():
    model = TinyLinearModel()

    report = convert_model(model, ConvertConfig(include=["q_proj"]))

    assert report.converted == ["q_proj"]
    assert isinstance(model.q_proj, TernaryBitLinear)
    assert isinstance(model.k_proj, nn.Linear)
    assert isinstance(model.other, nn.Linear)


def test_bs_plugin_targets_skips_non_divisible_and_is_idempotent():
    model = TinyPluginModel()
    model.q_proj.weight.requires_grad = False
    cfg = DictDefault(
        {
            "bs_qat": True,
            "bs_qat_mode": "ternary",
            "bs_qat_group_size": 128,
            "bs_qat_act_quant": False,
            "bs_qat_target_names": ["q_proj", "bad_proj", "lm_head"],
            "bs_qat_skip_names": ["lm_head"],
        }
    )

    plugin = BSQATPlugin()
    plugin.post_lora_load(cfg, model)
    plugin.post_model_load(cfg, model)

    assert isinstance(model.q_proj, BitLinear)
    assert model.q_proj.weight.requires_grad is False
    assert isinstance(model.bad_proj, nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert sum(isinstance(module, BitLinear) for module in model.modules()) == 1


def test_bs_internal_qat_lora_patches_before_peft_and_freezes_base():
    model = TinyPluginModel()
    cfg = DictDefault(
        {
            "adapter": None,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "bs_qat": True,
            "bs_qat_mode": "ternary",
            "bs_qat_group_size": 128,
            "bs_qat_act_quant": False,
            "bs_qat_lora_mode": "internal_qat",
            "bs_qat_freeze_non_lora": True,
            "bs_qat_target_names": ["q_proj"],
            "bs_qat_skip_names": ["lm_head"],
        }
    )

    BSQATPlugin().post_model_build(cfg, model)

    assert isinstance(model.q_proj, BitLinear)
    assert model.q_proj.use_lora is True
    assert model.q_proj.weight.requires_grad is False
    assert model.q_proj.lora_A.requires_grad is True
    assert model.q_proj.lora_B.requires_grad is True
    assert model.bad_proj.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False


def test_bs_internal_qat_rejects_external_peft_adapter():
    cfg = DictDefault(
        {
            "adapter": "lora",
            "bs_qat": True,
            "bs_qat_lora_mode": "internal_qat",
        }
    )

    try:
        BSQATPlugin().post_model_build(cfg, TinyPluginModel())
    except ValueError as exc:
        assert "incompatible with Axolotl `adapter: lora/qlora`" in str(exc)
    else:
        raise AssertionError("internal_qat must reject external PEFT adapters")


def test_bs_full_weight_patches_all_targets_without_freezing():
    model = TinyPluginModel()
    cfg = DictDefault(
        {
            "adapter": None,
            "bs_qat": True,
            "bs_qat_mode": "ternary",
            "bs_qat_group_size": 128,
            "bs_qat_act_quant": False,
            "bs_qat_lora_mode": "full_weight",
            "bs_qat_target_names": ["q_proj"],
            "bs_qat_skip_names": ["lm_head"],
        }
    )

    BSQATPlugin().post_model_load(cfg, model)

    assert isinstance(model.q_proj, BitLinear)
    assert model.q_proj.use_lora is False
    assert model.q_proj.lora_A is None
    # Unlike internal_qat, full_weight never freezes anything: the shadow weight
    # itself is what's trained, and untouched modules keep their original state.
    assert model.q_proj.weight.requires_grad is True
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.requires_grad is True


def test_bs_full_weight_rejects_adapter():
    cfg = DictDefault(
        {
            "adapter": "lora",
            "bs_qat": True,
            "bs_qat_lora_mode": "full_weight",
        }
    )

    try:
        BSQATPlugin().post_model_load(cfg, TinyPluginModel())
    except ValueError as exc:
        assert "bs_qat_lora_mode=full_weight is incompatible" in str(exc)
    else:
        raise AssertionError("full_weight must reject a PEFT adapter")


def test_load_full_checkpoint_state_dict_loads_model_safetensors(tmp_path):
    src = nn.Linear(8, 4, bias=False)
    dst = nn.Linear(8, 4, bias=False)
    save_file({"weight": src.weight.detach().clone()}, str(tmp_path / "model.safetensors"))

    info = load_full_checkpoint_state_dict(dst, tmp_path)

    assert info["loaded_tensors"] == 1
    assert torch.equal(src.weight, dst.weight)


def test_q2_0_pack_unpack_matches_training_dequantization():
    weights = torch.randn(3, 128)
    codes, scales = quantize_ternary(weights)

    packed = pack_q2_0(codes.numpy(), scales.numpy())
    unpacked = unpack_q2_0(packed, 128)
    dequantized = dequantize(codes, scales).numpy()

    assert packed.shape == (3, 34)
    assert np.max(np.abs(unpacked - dequantized)) == 0.0


def test_bitlinear_caches_frozen_base_fake_quant_weight(monkeypatch):
    layer = BitLinear(128, 64, bias=False)
    layer.weight.requires_grad = False
    calls = 0
    original_fake_quantize = bitlinear_mod.fake_quantize

    def counting_fake_quantize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_fake_quantize(*args, **kwargs)

    monkeypatch.setattr(bitlinear_mod, "fake_quantize", counting_fake_quantize)

    x = torch.randn(2, 128)
    layer(x)
    layer(x)

    assert calls == 1

    with torch.no_grad():
        layer.weight.add_(0.01)
    layer(x)

    assert calls == 2


def test_bitlinear_does_not_cache_trainable_qat_weight(monkeypatch):
    layer = BitLinear(128, 64, bias=False)
    calls = 0
    original_fake_quantize = bitlinear_mod.fake_quantize

    def counting_fake_quantize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_fake_quantize(*args, **kwargs)

    monkeypatch.setattr(bitlinear_mod, "fake_quantize", counting_fake_quantize)

    x = torch.randn(2, 128)
    layer(x)
    layer(x)

    assert calls == 2


def test_bitlinear_internal_lora_fake_quantizes_merged_weight(monkeypatch):
    layer = BitLinear(
        128,
        16,
        bias=False,
        use_lora=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    captured = []
    original_fake_quantize = bitlinear_mod.fake_quantize

    def capturing_fake_quantize(w, *args, **kwargs):
        captured.append(w.detach().clone())
        return original_fake_quantize(w, *args, **kwargs)

    monkeypatch.setattr(bitlinear_mod, "fake_quantize", capturing_fake_quantize)

    x = torch.randn(2, 128)
    layer.eval()
    layer(x)

    expected = layer.weight.detach().float() + (layer.lora_B @ layer.lora_A).detach() * layer.lora_scaling
    assert len(captured) == 1
    assert torch.allclose(captured[0], expected)


def test_internal_qat_lora_save_load_roundtrip(tmp_path):
    src = BitLinear(128, 16, bias=False, use_lora=True, lora_r=4, lora_alpha=8)
    dst = BitLinear(128, 16, bias=False, use_lora=True, lora_r=4, lora_alpha=8)
    with torch.no_grad():
        src.lora_A.fill_(0.123)
        src.lora_B.fill_(-0.456)

    info = save_internal_lora(src, tmp_path)
    load_info = load_internal_lora(dst, tmp_path)

    assert info["tensors"] == 2
    assert load_info["loaded_tensors"] == 2
    assert torch.equal(src.lora_A, dst.lora_A)
    assert torch.equal(src.lora_B, dst.lora_B)


def test_internal_qat_lora_callback_saves_checkpoint_adapter(tmp_path):
    model = BitLinear(128, 16, bias=False, use_lora=True, lora_r=4, lora_alpha=8)
    trainer = SimpleNamespace(model=model, accelerator=None)
    cfg = DictDefault(
        {
            "bs_qat": True,
            "bs_qat_lora_mode": "internal_qat",
        }
    )
    callback = BSQATPlugin().add_callbacks_post_trainer(cfg, trainer)[0]

    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=12, is_world_process_zero=True),
        SimpleNamespace(),
    )

    assert (tmp_path / "checkpoint-12" / "internal_qat_lora" / ADAPTER_WEIGHTS).exists()


def test_internal_qat_lora_post_train_saves_final_adapter(tmp_path):
    model = BitLinear(128, 16, bias=False, use_lora=True, lora_r=4, lora_alpha=8)
    cfg = DictDefault(
        {
            "output_dir": str(tmp_path),
            "bs_qat": True,
            "bs_qat_lora_mode": "internal_qat",
        }
    )

    BSQATPlugin().post_train(cfg, model)

    assert (tmp_path / "internal_qat_lora" / ADAPTER_WEIGHTS).exists()


def test_internal_qat_lora_post_train_skips_non_zero_rank(tmp_path, monkeypatch):
    # Regression test: axolotl's train() calls cleanup_distributed() before firing
    # the post_train hook, so on non-zero ranks LOG.info (via is_main_process() ->
    # dist.get_rank()) used to raise "Default process group has not been
    # initialized" even after a fully successful run -- see the crash observed at
    # the end of both bonsai_internal_qat_non_reasoning{,_v2} train.log runs.
    monkeypatch.setenv("RANK", "1")
    model = BitLinear(128, 16, bias=False, use_lora=True, lora_r=4, lora_alpha=8)
    cfg = DictDefault(
        {
            "output_dir": str(tmp_path),
            "bs_qat": True,
            "bs_qat_lora_mode": "internal_qat",
        }
    )

    BSQATPlugin().post_train(cfg, model)

    assert not (tmp_path / "internal_qat_lora" / ADAPTER_WEIGHTS).exists()


def test_matches_target_segment_excludes_fused_projection():
    # "up_proj" must match a real up_proj module but not a fused gate_up_proj one
    # (some architectures merge gate/up into a single linear with that name).
    assert plugin_mod._matches_target("model.layers.0.mlp.up_proj", "up_proj") is True
    assert plugin_mod._matches_target("model.layers.0.mlp.gate_up_proj", "up_proj") is False
    assert plugin_mod._matches_target("mlp.gate_proj", "mlp.gate_proj") is True
