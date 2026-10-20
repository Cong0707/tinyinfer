"""Speculative-verification operators for V4.1: tree attention, rejection sampling, DSpark interface.

Interfaces (to be wired into tinyinfer decode loop):
  - dspark_idxs(start_pos, window, block_size)         -> [1, block, topk] int32 (official DSpark layout)
  - tree_idxs(start_pos, window, parents)              -> [1, m, window+m] int32 (branching draft tree)
  - verify_attention(q, kv, sink, idxs, scale)         -> sparse_attn_sm120 (exact)
  - rejection_sample(draft_probs, target_probs, ids)   -> accepted prefix + next token
  - tree_rejection(target_probs, draft_probs, parents, ids) -> best path
  - MockDSparkHead(hidden, prev_token)                  -> draft ids + confidence (cost model)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def dspark_idxs(start_pos: int, window: int, block_size: int, device="cuda") -> torch.Tensor:
    """Official get_dspark_topk_idxs: window slots + all block slots, same row for every draft token."""
    assert start_pos > 0
    win = torch.arange(min(window, start_pos + 1), device=device)
    blk = window + torch.arange(block_size, device=device)
    row = torch.cat([win, blk]).int()
    return row.view(1, 1, -1).expand(1, block_size, -1).contiguous()


def tree_idxs(start_pos: int, window: int, parents: list[int], device="cuda") -> torch.Tensor:
    """Branching draft tree: each draft token attends the sliding window + itself + its ancestors.

    parents[i] = index of i's parent inside the draft block, or -1 for a root. KV layout:
    [0:window] window ring, [window:window+m] draft tokens.
    """
    m = len(parents)
    if start_pos + 1 <= window:
        wins = torch.arange(0, start_pos + 1, device=device)
    else:
        oldest = start_pos % window + 1
        wins = torch.cat([torch.arange(oldest, window, device=device), torch.arange(0, oldest, device=device)])
    topk = min(window, start_pos + 1) + m
    out = torch.full((1, m, topk), -1, dtype=torch.int32, device=device)
    for i in range(m):
        chain = [window + i]
        p = parents[i]
        while p >= 0:
            chain.append(window + p)
            p = parents[p]
        row = torch.cat([wins, torch.tensor(sorted(chain), device=device, dtype=torch.int64)])
        out[0, i, : row.numel()] = row.int()
    return out


def ref_attention(q, kv, sink, idxs, scale):
    """Naive gather reference matching the sparse_attn convention (sink as extra logit)."""
    b, s, h, d = q.shape
    t = idxs.shape[-1]
    valid = idxs >= 0
    k = kv.gather(1, idxs.clamp(min=0).reshape(b, s * t, 1).expand(b, s * t, d)).reshape(b, s, t, d)
    sc = torch.einsum("bshd,bstd->bsht", q.float(), k.float()) * scale
    sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = torch.maximum(sc.max(-1).values, sink.view(1, 1, h))
    p = torch.exp(sc - mx.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(sink.view(1, 1, h) - mx)
    return torch.einsum("bsht,bstd->bshd", p / den.unsqueeze(-1), k.float()).to(q.dtype)


def verify_attention(q, kv, sink, idxs, scale, exp_mode="exact"):
    import sys
    sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\prototypes\kernels")
    from sparse_attn_sm120 import sparse_attn_sm120
    return sparse_attn_sm120(q, kv, sink, idxs, scale, exp_mode=exp_mode)


# ---------------- rejection sampling ----------------

def rejection_sample(draft_probs: torch.Tensor, target_probs: torch.Tensor, draft_ids: torch.Tensor,
                     target_next: torch.Tensor | None = None, generator=None):
    """Speculative sampling (Leviathan et al.) for a linear draft of K tokens.

    draft_probs:  [B, K, V]  (draft model distributions for the K draft tokens)
    target_probs: [B, K, V]  (target distributions at the same positions)
    draft_ids:    [B, K]     (draft token ids)
    target_next:  [B, V]     (target distribution after the last draft token; for the all-accepted case)
    Returns (accepted_len [B], tokens [B, K+1]).
    """
    B, K, V = draft_probs.shape
    dev = draft_probs.device
    ids = draft_ids.long()
    q = draft_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30)
    p = target_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    u = torch.rand(q.shape, device=dev, generator=generator)
    accept = u < (p / q).clamp(max=1.0)
    first_rej = torch.where(accept.all(-1), torch.full_like(ids[:, 0], K), (~accept).float().argmax(-1))
    ar = torch.arange(B, device=dev)
    # residual distribution at the first rejected position
    resid = (target_probs.clamp_min(0) - draft_probs.clamp_min(0)).clamp_min(0)
    resid = resid[ar, first_rej.clamp_max(K - 1)]                      # [B, V]
    if target_next is not None:
        last = target_next.clamp_min(0)
        resid = torch.where((first_rej == K).unsqueeze(-1), last, resid)
    resid = resid / resid.sum(-1, keepdim=True).clamp_min(1e-30)
    cdf = resid.cumsum(-1)
    uu = torch.rand((B, 1), device=dev, generator=generator) * cdf[:, -1:]
    next_id = torch.searchsorted(cdf, uu).squeeze(-1).clamp_max(V - 1)   # [B]
    tokens = torch.empty(B, K + 1, dtype=torch.long, device=dev)
    tokens[:, :K] = ids
    # positions < first_rej keep the draft token, first_rej gets the sampled token
    for b in range(B):
        tokens[b, first_rej[b]] = next_id[b]
    return first_rej, tokens


def tree_rejection(target_probs: torch.Tensor, draft_probs: torch.Tensor, parents: list[int],
                   draft_ids: torch.Tensor, target_next: torch.Tensor | None = None, generator=None):
    """Run linear rejection on every root->leaf path of a small tree, keep the longest accepted path.

    target_probs/draft_probs: [B, m, V] for the m draft nodes; draft_ids [B, m].
    Returns (best_path [B, m_best], accepted_len [B]).
    """
    import numpy as np
    m = len(parents)
    children = [[] for _ in range(m)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    paths = []
    def dfs(node, path):
        path = path + [node]
        if not children[node]:
            paths.append(path)
        for c in children[node]:
            dfs(c, path)
    for i, p in enumerate(parents):
        if p < 0:
            dfs(i, [])
    B = draft_ids.shape[0]
    best_len = torch.full((B,), -1, dtype=torch.long)
    best_tokens = torch.zeros(B, m, dtype=torch.long)
    ar = torch.arange(B)
    for path in paths:
        idx = torch.tensor(path, device=draft_ids.device)
        al, toks = rejection_sample(draft_probs[:, idx], target_probs[:, idx], draft_ids[:, idx], target_next, generator)
        better = al > best_len
        best_len = torch.where(better, al, best_len)
        best_tokens[better] = F.pad(toks[:, : len(path)], (0, m - len(path)))
    return best_tokens, best_len


# ---------------- DSpark cost model / interface ----------------

class MockMarkovHead(nn.Module):
    """DSparkMarkovHead: token -> rank embedding -> vocab logits (bias for the main head)."""
    def __init__(self, vocab: int = 129280, rank: int = 256, device="cuda"):
        super().__init__()
        self.embed = nn.Embedding(vocab, rank).to(device=device, dtype=torch.bfloat16)
        self.head = nn.Linear(rank, vocab, bias=False).to(device=device, dtype=torch.bfloat16)

    def forward(self, token_ids):  # [B] -> logits [B,V], embed [B,rank]
        emb = self.embed(token_ids)
        return self.head(emb), emb


class MockConfidenceHead(nn.Module):
    def __init__(self, dim: int = 5120, rank: int = 256, device="cuda"):
        super().__init__()
        self.proj = nn.Linear(dim + rank, 1).to(device=device, dtype=torch.float32)

    def forward(self, hidden, markov_embed):  # [B,m,dim],[B,m,rank] -> [B,m]
        return self.proj(torch.cat([hidden.float(), markov_embed.float()], -1)).squeeze(-1)


def dspark_block(hidden, prev_token, markov: MockMarkovHead, confidence: MockConfidenceHead,
                 block_size: int = 5, vocab: int = 129280, temperature: float = 1.0):
    """One DSpark draft block: autoregressively sample block_size tokens with the markov head,
    then score the whole block with the confidence head. Matches the official loop shape."""
    B = hidden.shape[0]
    hidden_block = hidden.unsqueeze(1).expand(-1, block_size, -1).contiguous() if hidden.dim() == 2 else hidden
    embeds = []
    out_ids = torch.empty(B, block_size + 1, dtype=torch.long, device=hidden.device)
    out_ids[:, 0] = prev_token
    logits_all = []
    for i in range(block_size):
        bias, emb = markov(out_ids[:, i])
        embeds.append(emb)
        logits = bias.float() / temperature
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
        out_ids[:, i + 1] = nxt
        logits_all.append(logits)
    conf = confidence(hidden_block, torch.stack(embeds, dim=1))
    return out_ids, torch.stack(logits_all, dim=1), conf


def plan_verify(conf: torch.Tensor, threshold: float = 0.5, max_block: int = 5) -> torch.Tensor:
    """Confidence gating: number of draft tokens to verify per request (DSpark-style scheduling)."""
    keep = (conf >= threshold).sum(-1) + 1
    return keep.clamp(max=max_block)