"""Process-wide RoPE tables.

The rotation table is a model constant (1M positions x rotary_dim complex), not a
per-layer weight. Keeping a copy per attention module made every streamed layer
carry 256MB of identical bytes: 36 layers x 256MB = 9.2GB per token of pure
duplication. This module hands out one tensor per (dim, base, seqlen, yarn args).
"""
import torch

_TABLES: dict = {}


def get_table(key, factory, device=None):
    """One tensor per key; moved to *device* once and shared by every caller."""
    t = _TABLES.get(key)
    if t is None:
        t = factory()
        _TABLES[key] = t
    # tables are model constants but every layer reads them on the model device.
    # Resolving through here (rather than .to() in each module) keeps ONE copy: a
    # per-instance .to() gave all 40 layers their own 256MB table (10GB -> paging).
    if device is not None and t.device != torch.device(device):
        t = t.to(device)
        _TABLES[key] = t
    return t


def table_key(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    return (int(dim), int(seqlen), int(original_seq_len), float(base),
            float(factor), float(beta_fast), float(beta_slow))
