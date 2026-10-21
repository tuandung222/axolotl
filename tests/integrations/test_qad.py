"""CPU tests: grid parity, pack/unpack round trip, chunked KL == naive KL. Run: PYTHONPATH=src python tests/test_qad_basics.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axolotl.integrations.qad.fixed_scale_fq import QMAX, QMIN, FQLinear, add_internal_lora, apply_fixed_scale_fq, fake_quant, fake_quant_lsq  # noqa: E402
from axolotl.integrations.qad.kl import chunked_kl_loss  # noqa: E402

torch.manual_seed(0)
GS = 128

def test_pack_roundtrip():
    try:
        from compressed_tensors.compressors.pack_quantized import pack_to_int32, unpack_from_int32
    except ImportError:
        print("  [skip test_pack_roundtrip: compressed_tensors.compressors.pack_quantized not installed]")
        return
    q = torch.randint(QMIN, QMAX + 1, (64, 256), dtype=torch.int8)
    assert torch.equal(unpack_from_int32(pack_to_int32(q, 4), 4, torch.Size(q.shape)), q)

def test_fixed_grid_is_idempotent_and_dynamic_grid_is_not():
    w = torch.randn(32, 256)
    s = (w.view(32, 2, GS).abs().amax(-1) / 7.5).to(torch.bfloat16)
    wq = fake_quant(w.to(torch.bfloat16), s, GS)
    assert torch.equal(fake_quant(wq, s, GS), wq), "fixed-scale fake-quant must be idempotent"
    dyn = (wq.float().view(32, 2, GS).abs().amax(-1) / 7.5).to(torch.bfloat16)  # torchao-style recompute
    assert not torch.equal(dyn, s), "expected the recomputed grid to differ (documents why scales are fixed)"
    print("  frac groups whose recomputed scale != PTQ scale:", (dyn != s).float().mean().item())

def test_ste_gradient_and_module_swap():
    lin = torch.nn.Linear(256, 8, bias=False).to(torch.bfloat16)
    s = (lin.weight.float().view(8, 2, GS).abs().amax(-1) / 7.5).to(torch.bfloat16)
    m = torch.nn.Sequential(lin)
    assert apply_fixed_scale_fq(m, {"0": s}, GS) == ["0"] and isinstance(m[0], FQLinear)
    assert "fq_scale" not in m.state_dict()
    m(torch.randn(4, 256, dtype=torch.bfloat16)).sum().backward()
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0

def test_chunked_kl_matches_naive():
    B, T, H, V = 2, 37, 16, 50
    hs, ht = torch.randn(B, T, H, requires_grad=True), torch.randn(B, T, H)
    ws, wt = torch.randn(V, H, requires_grad=True), torch.randn(V, H)
    labels = torch.randint(0, V, (B, T)); labels[:, :9] = -100; labels[0, 20:25] = -100
    tot, n = chunked_kl_loss(hs, ht, ws, wt, labels, chunk=7)
    sel = labels[:, 1:] != -100
    ls, lt = (hs[:, :-1] @ ws.t())[sel], (ht[:, :-1] @ wt.t())[sel]
    pt = lt.log_softmax(-1); ref = (pt.exp() * (pt - ls.log_softmax(-1))).sum()
    assert n == int(sel.sum()) and torch.allclose(tot, ref, rtol=1e-5), (tot.item(), ref.item())
    tot.backward(); assert hs.grad is not None and ws.grad is not None

def test_topk_modes_match_naive():
    B, T, H, V, K = 2, 33, 16, 60, 10
    hs, ht = torch.randn(B, T, H, requires_grad=True), torch.randn(B, T, H)
    ws, wt = torch.randn(V, H, requires_grad=True), torch.randn(V, H)
    labels = torch.randint(0, V, (B, T)); labels[:, :6] = -100
    sel = labels[:, 1:] != -100
    ls, lt = (hs[:, :-1] @ ws.t())[sel], (ht[:, :-1] @ wt.t())[sel]
    idx = lt.topk(K, -1).indices
    lpt, lps = lt.log_softmax(-1), ls.log_softmax(-1)
    ref_trunc = (lpt.gather(-1, idx).exp() * (lpt.gather(-1, idx) - lps.gather(-1, idx))).sum()
    a, b = lt.gather(-1, idx).log_softmax(-1), ls.gather(-1, idx).log_softmax(-1)
    ref_renorm = (a.exp() * (a - b)).sum()
    full, _ = chunked_kl_loss(hs, ht, ws, wt, labels, chunk=5)
    tr, _ = chunked_kl_loss(hs, ht, ws, wt, labels, chunk=5, topk=K, topk_mode="truncated")
    rn, _ = chunked_kl_loss(hs, ht, ws, wt, labels, chunk=5, topk=K, topk_mode="renorm")
    assert torch.allclose(tr, ref_trunc, rtol=1e-5) and torch.allclose(rn, ref_renorm, rtol=1e-5)
    assert tr < full + 1e-6, "top-k truncated KL is a partial sum of non-negative... terms of the full KL? (not guaranteed per-term)"
    tr.backward(); assert hs.grad is not None

def test_lsq_forward_equals_fixed_and_scale_gets_gradient():
    w = torch.randn(16, 256).to(torch.bfloat16); s = (w.float().view(16, 2, GS).abs().amax(-1) / 7.5).to(torch.bfloat16)
    assert torch.equal(fake_quant_lsq(w, s, GS), fake_quant(w, s, GS)), "LSQ forward must equal the fixed-scale forward"
    lin = torch.nn.Linear(256, 16, bias=False).to(torch.bfloat16); m = torch.nn.Sequential(lin)
    apply_fixed_scale_fq(m, {"0": (lin.weight.float().view(16, 2, GS).abs().amax(-1) / 7.5).to(torch.bfloat16)}, GS, learn_scale=True)
    assert lin.fq_logs.shape == (16, 2) and float(lin.fq_logs.abs().sum()) == 0.0
    m(torch.randn(4, 256, dtype=torch.bfloat16)).float().pow(2).sum().backward()
    assert lin.fq_logs.grad is not None and float(lin.fq_logs.grad.abs().sum()) > 0 and lin.weight.grad is not None
    assert any(n.endswith("fq_logs") for n, _ in m.named_parameters())

def test_internal_lora_init_equals_rtn_and_merge_matches_forward():
    torch.manual_seed(1)
    lin = torch.nn.Linear(256, 16, bias=False).to(torch.bfloat16); m = torch.nn.Sequential(lin)
    s = (lin.weight.float().view(16, 2, GS).abs().amax(-1) / 7.5).to(torch.bfloat16)
    w0 = lin.weight.detach().clone(); x = torch.randn(4, 256, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, fake_quant(w0, s, GS))
    apply_fixed_scale_fq(m, {"0": s}, GS); add_internal_lora(m, ["0"], 8, 16.0)
    assert torch.equal(m(x), ref), "B=0 must reproduce the RTN model exactly"
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {"0.lora_A", "0.lora_B"}
    with torch.no_grad(): lin.lora_B.normal_(0, 0.05)
    out = m(x); (out.float() ** 2).sum().backward()
    assert lin.lora_B.grad is not None and float(lin.lora_B.grad.abs().sum()) > 0 and lin.weight.grad is None
    delta = (lin.lora_B.float() @ lin.lora_A.float()) * float(lin.lora_scale)           # export-side merge
    merged = fake_quant((lin.weight.float() + delta), s, GS).to(torch.bfloat16)
    assert torch.equal(torch.nn.functional.linear(x, merged), out), "exported (merged+rounded) weights must equal the training forward"

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
