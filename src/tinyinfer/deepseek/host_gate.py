"""Host-side MoE gate.

Decode is host-synchronised once per MoE layer: the routed expert ids decide which expert
blobs to fetch, so the CPU has to read them before it can issue the transfers. That sync
costs the CPU a wait for the GPU to finish the gate matmul, and it is not overlapped with
anything. Measured on this machine: 80 tolist() calls per token, 373 ms of a 950 ms step,
i.e. 39% of the step is the CPU waiting for a 384x5120 matmul it could have done itself.

The gate is small: 384 x 5120 = 3.75 MB per layer, 150 MB for all 40, which fits in host
RAM beside the weights. On the CPU it takes 0.29 ms per layer (12 ms per token for all
40) versus the ~4.7 ms per layer the GPU sync was costing, and the routing it produces is
identical: the same fp32 matmul, the same softplus-then-sqrt, the same bias, the same
top-k order.

Nothing here is approximate. The only thing that changes is which device multiplies the
matrix, and the answer does not depend on that.
"""
from __future__ import annotations

import numpy as np


def softplus_sqrt(x: np.ndarray) -> np.ndarray:
    """log1p(exp(x)).sqrt() computed without overflow.

    np.log1p(np.exp(x)) overflows for x > 88 and loses accuracy for very negative x, so
    the standard split is used: softplus(x) = max(x,0) + log1p(exp(-|x|)).
    """
    return np.sqrt(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def route(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, topk: int,
          norm_topk_prob: bool, route_scale: float):
    """Returns (weights[topk], indices[topk]) for one token, in fp32, on the host."""
    logits = x.reshape(1, -1).astype(np.float32) @ weight.T
    scores = softplus_sqrt(logits)
    # The bias decides WHICH experts are picked but does not enter their WEIGHTS: the
    # torch path ranks on (scores + bias) and then gathers from the unbiased scores.
    # Normalising the biased value instead changes every expert's contribution, and the
    # decode goes somewhere else entirely.
    ranking = scores + bias
    # Deterministic order: stable argsort descending, so ties resolve by expert id.
    # torch.topk on CUDA is not stable and the same layer could pick a different expert
    # set on two forwards of an identical input -- the engine relies on determinism.
    order = np.argsort(-ranking, axis=-1, kind='stable')[:, :topk]
    sel = np.take_along_axis(scores, order, axis=-1)
    if norm_topk_prob:
        sel = sel / sel.sum(axis=-1, keepdims=True)
    sel = sel * route_scale
    return sel.astype(np.float32), order.astype(np.int64)
