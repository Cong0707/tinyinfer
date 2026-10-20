"""Fast config sweep: load once, warm up JIT, then time decode across small variants."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

MODEL = str(ROOT / "cache" / "weights")
LAYERS = [0, 1, 2, 3, 4]


def build(gpu_layers=None, slots=48, head_cpu=True):
    eng = Engine(MODEL, tier=OptLevel.BALANCED, device="cuda", layers=LAYERS, gpu_layers=gpu_layers)
    if eng.expert_store is not None and slots != 48:
        # rebuild the store with the requested slot count
        eng.expert_store.close()
        from tinyinfer.loading import ExpertStore
        eng.expert_store = ExpertStore.from_store(eng.store, cache_bytes=8 << 30, policy="arc",
                                                  vram_slots=slots, device="cuda", direct=False)
        for b in eng._blocks.values():
            b.ffn._expert_store = eng.expert_store
    if not head_cpu:
        eng.head_on_cpu = False
        eng.head = eng.head.to("cuda")
    return eng


def timeit(eng, prompt, decode=2, repeat=2):
    best = None
    for _ in range(repeat):
        eng.reset()
        with torch.inference_mode():
            logits = eng.forward(torch.tensor([prompt]), 0)
            pos = len(prompt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _s in range(decode):
                nid = int(logits[0, -1].argmax())
                logits = eng.forward(torch.tensor([[nid]]), pos)
                pos += 1
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / decode * 1e3
        best = dt if best is None else min(best, dt)
    return best


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = ([0] + tok("The quick brown fox jumps over the lazy dog.").input_ids)[:16]

    configs = [
        ("base_ramprefetch", dict(gpu_layers=25, slots=48, head_cpu=False)),
        ("prefetch_off", dict(gpu_layers=25, slots=48, head_cpu=False), {"prefetch": False}),
        ("slots64", dict(gpu_layers=25, slots=64, head_cpu=False)),
    ]
    print(f"{'config':18s} {'decode ms':>10s} {'tok/s':>7s} {'vram MB':>8s}")
    for cfg in configs:
        name, kw = cfg[0], cfg[1]
        extra = cfg[2] if len(cfg) > 2 else {}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        eng = build(**kw)
        if "prefetch" in extra:
            eng.enable_prefetch = extra["prefetch"]
        if "cache_wo_a" in extra:
            eng.cache_wo_a = extra["cache_wo_a"]
        with torch.inference_mode():
            eng.forward(torch.tensor([prompt]), 0)  # JIT warmup
        ms = timeit(eng, prompt, repeat=3)
        print(f"{name:18s} {ms:10.1f} {1000/ms:7.2f} {torch.cuda.max_memory_allocated()//2**20:8d}")


if __name__ == "__main__":
    main()
