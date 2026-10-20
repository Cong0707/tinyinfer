"""Self-test for the GGUF prototype.

Checks:
1. parse a real llama.cpp vocabulary GGUF (no model weights needed);
2. generate a synthetic GGUF and read it back through the same parser;
3. compare every dequantizer against gguf-py's reference implementation;
4. verify the HF -> ggml MXFP4 repack is bit-lossless.

Run:
    python selftest.py
    LLAMA_CPP_DIR=/path/to/llama.cpp python selftest.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
TINYINFER = HERE.parents[3]
WORKSPACE = HERE.parents[4]
LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR", WORKSPACE / "repositories" / "llama.cpp"))

try:
    from . import gguf_reader, make_sample_gguf, quants_ref
except ImportError:  # direct script execution
    sys.path.insert(0, str(HERE.parent))
    import gguf_reader  # type: ignore
    import make_sample_gguf  # type: ignore
    import quants_ref  # type: ignore


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_real_vocab() -> None:
    path = LLAMA_CPP / "models" / "ggml-vocab-deepseek-llm.gguf"
    if not path.is_file():
        print(f"[skip] real vocab GGUF not found: {path}")
        return
    with gguf_reader.GGUFReader(path) as r:
        _check(r.version == 3, f"unexpected GGUF version {r.version}")
        _check(r.n_kv > 10, f"too few metadata keys: {r.n_kv}")
        tokens = r.metadata.get("tokenizer.ggml.tokens")
        _check(isinstance(tokens, list) and len(tokens) > 1000, "tokenizer.ggml.tokens missing")
        _check(r.metadata.get("tokenizer.ggml.model") in ("gpt2", "llama"), "unexpected tokenizer model")
        arch = r.metadata.get("general.architecture")
        print(f"[ok] real GGUF: {path.name}: version={r.version}, "
              f"kv={r.n_kv}, tensors={r.n_tensors}, tokens={len(tokens)}, arch={arch!r}, "
              f"alignment={r.alignment}")


def test_synthetic() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.gguf"
        make_sample_gguf.build(path, seed=7)
        with gguf_reader.GGUFReader(path) as r:
            _check(r.version == 3, "synthetic version")
            _check(r.metadata["general.architecture"] == "deepseek4", "architecture")
            _check(r.metadata["deepseek4.attention.compress_ratios"] == [0, 4], "array KV")
            _check(r.metadata["tinyinfer.proto.bool"] is True, "bool KV")
            _check(abs(r.metadata["tinyinfer.proto.float"] - 3.5) < 1e-6, "float KV")
            _check(len(r.metadata["tokenizer.ggml.tokens"]) == 4, "string array KV")
            _check(r.n_tensors == 5, "tensor count")

            t = r.get("token_embd.weight")
            _check(t.type_name == "Q4_K", "Q4_K type")
            _check(t.dims == (256, 4), "Q4_K dims")
            _check(t.shape == (4, 256), "logical shape")
            _check(t.n_bytes == 4 * 144, "Q4_K nbytes")
            y = t.dequantize()
            _check(y.shape == (4, 256), "Q4_K dequant shape")
            _check(np.isfinite(y).all(), "Q4_K dequant finite")

            e = r.get("blk.0.ffn_gate_exps.weight")
            _check(e.type_name == "MXFP4", "MXFP4 type")
            _check(e.dims == (32, 16, 4), "MXFP4 dims")
            _check(e.shape == (4, 16, 32), "MXFP4 logical shape")
            _check(e.n_bytes == 64 * 17, "MXFP4 nbytes")
            _check(np.isfinite(e.dequantize()).all(), "MXFP4 dequant finite")

            q6 = r.get("blk.0.ffn_down_exps.weight")
            _check(q6.dequantize().shape == (2, 8, 256), "Q6_K shape")
            _check((q6.data_offset - r.data_offset) % r.alignment == 0, "tensor alignment")

            iq = r.get("blk.1.ffn_up_exps.weight")
            _check(iq.type_name == "IQ2_XXS", "IQ2_XXS header type")
            _check(iq.n_bytes == 8 * 66, "IQ2_XXS nbytes")
            try:
                iq.dequantize()
            except NotImplementedError:
                pass
            else:
                raise AssertionError("IQ2_XXS should be header-only in this prototype")
            print("[ok] IQ2_XXS header parsed (dequantization intentionally not implemented)")
        print("[ok] synthetic GGUF: metadata, tensor directory, Q4_K/Q6_K/MXFP4 dequant")


def _reference_dequant() -> None:
    """Import gguf-py from the llama.cpp checkout (optional test oracle)."""
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf  # type: ignore
    from gguf import quants  # type: ignore

    rng = np.random.default_rng(123)
    sizes = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "Q8_0": 34, "MXFP4": 17}
    one = np.array([1.0], dtype=np.float16).view(np.uint8)
    zero = np.array([0.0], dtype=np.float16).view(np.uint8)
    for name, type_size in sizes.items():
        blocks = rng.integers(0, 256, size=(8, type_size), dtype=np.uint8)
        if name in ("Q4_K", "Q5_K"):
            blocks[:, 0:2] = one      # d
            blocks[:, 2:4] = zero     # dmin
        elif name == "Q6_K":
            blocks[:, 208:210] = one  # d
        elif name == "Q8_0":
            blocks[:, 0:2] = one      # d
        elif name == "MXFP4":
            blocks[:, 0] = 127        # E8M0: 2^0
        ours = quants_ref.dequantize(blocks, name)
        ref = quants.dequantize(blocks, getattr(gguf.GGMLQuantizationType, name))
        _check(ours.shape == ref.shape, f"{name}: shape {ours.shape} != {ref.shape}")
        exact = np.array_equal(ours, ref, equal_nan=True)
        _check(exact, f"{name}: dequant differs from gguf-py")
        print(f"[ok] {name}: bit-exact vs gguf-py, shape={ours.shape}")


def test_reference_dequant() -> None:
    if not (LLAMA_CPP / "gguf-py" / "gguf" / "quants.py").is_file():
        print(f"[skip] gguf-py not found under {LLAMA_CPP}")
        return
    try:
        _reference_dequant()
    except ImportError as e:
        print(f"[skip] cannot import gguf-py: {e}")


def test_mxfp4_repack() -> None:
    rng = np.random.default_rng(5)
    rows, cols = 6, 256
    packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    scale = rng.integers(120, 135, size=(rows, cols // 32), dtype=np.uint8)
    blocks = quants_ref.repack_hf_mxfp4(packed, scale)
    _check(blocks.shape == (rows, (cols // 32) * 17), "repack shape")

    expected = quants_ref.hf_fp4_to_float(packed, scale)
    actual = quants_ref.dequantize(blocks, "MXFP4").reshape(rows, cols)
    _check(np.array_equal(expected, actual), "HF -> ggml MXFP4 repack is not lossless")
    print(f"[ok] MXFP4 repack: HF nibble order -> ggml blocks, "
          f"{rows}x{cols} values bit-identical")


def main() -> int:
    print(f"tinyinfer GGUF prototype selftest")
    print(f"  tinyinfer : {TINYINFER}")
    print(f"  llama.cpp : {LLAMA_CPP}")
    test_real_vocab()
    test_synthetic()
    test_reference_dequant()
    test_mxfp4_repack()
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
