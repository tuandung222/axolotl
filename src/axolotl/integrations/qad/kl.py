"""Chunked full-vocab KL(teacher || student) from final-norm hidden states + lm_head weights.

Never materialises [tokens, vocab] logits for the whole sequence: each chunk is recomputed in backward
(torch.utils.checkpoint). Vocab is 248k, so a 32k-token sequence in fp32 would be ~32 GB per tensor otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(hs, ht, ws, wt, labels, temperature, kl_alpha, ce_alpha, topk=None, topk_mode="truncated"):
    ls = (hs @ ws.t()).float()
    if topk is None:  # full-vocab KL(T || S)
        with torch.no_grad():
            lt = (ht @ wt.t()).float()
            logp_t = F.log_softmax(lt / temperature, dim=-1)
        logp_s = F.log_softmax(ls / temperature, dim=-1)
        kl = F.kl_div(logp_s, logp_t, reduction="sum", log_target=True)
    elif topk_mode == "truncated":  # teacher's top-k terms of the full KL; student log-probs keep the full-vocab
        with torch.no_grad():
            lt = (ht @ wt.t()).float()
            idx = lt.topk(topk, dim=-1).indices
            lt_k = F.log_softmax(lt / temperature, dim=-1).gather(-1, idx)
        ls_k = F.log_softmax(ls / temperature, dim=-1).gather(-1, idx)  # denominator, so mass leaking to the tail is penalised
        kl = F.kl_div(ls_k, lt_k, reduction="sum", log_target=True)
    elif topk_mode == "renorm":  # both distributions renormalised over the teacher's top-k tokens only (no full-vocab log_softmax!)
        with torch.no_grad():
            lt = (ht @ wt.t()).float()
            lt_top, idx = lt.topk(topk, dim=-1)
            lt_k = F.log_softmax(lt_top / temperature, dim=-1)
        ls_k = F.log_softmax(ls.gather(-1, idx) / temperature, dim=-1)
        kl = F.kl_div(ls_k, lt_k, reduction="sum", log_target=True)
    else:
        raise ValueError(f"unknown topk_mode {topk_mode}")
    kl = kl * (temperature**2)
    loss = kl_alpha * kl
    if ce_alpha:
        loss = loss + ce_alpha * F.cross_entropy(ls, labels, reduction="sum")
    return loss


def chunked_kl_loss(hs, ht, ws, wt, labels, chunk=1024, temperature=1.0, kl_alpha=1.0, ce_alpha=0.0, topk=None, topk_mode="truncated"):
    """hs/ht: [B,T,H] student/teacher final hidden; ws/wt: [V,H]; labels: [B,T] (-100 = ignore, UNshifted).
    Returns (sum_loss over supervised tokens, n_tokens). Position t predicts token t+1."""
    tgt = labels[:, 1:]
    sel = tgt != -100
    hs_s = hs[:, :-1][sel]
    ht_s = ht[:, :-1][sel]
    tgt = tgt[sel]
    n = int(tgt.numel())
    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        total = total + checkpoint(
            _chunk_loss, hs_s[sl], ht_s[sl], ws, wt, tgt[sl], temperature, kl_alpha, ce_alpha, topk, topk_mode, use_reentrant=False
        )
    return total, n
