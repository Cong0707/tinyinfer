"""Poll the checkpoint until all 40 layers are complete, then run the full self-check + bench.

Usage:
    python scripts/watch_ready.py --interval 300 --bench
Runs forever (Ctrl-C to stop); logs one line per check.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "cache" / "weights"))
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--max-wait-hours", type=float, default=24.0)
    args = ap.parse_args()

    py = sys.executable
    check = [py, str(ROOT / "scripts" / "full_check.py"), "--model-dir", args.model_dir]
    deadline = time.time() + args.max_wait_hours * 3600
    while time.time() < deadline:
        proc = subprocess.run(check, capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip().splitlines()
        msg = out[-1] if out else f"exit {proc.returncode}"
        print(f"[watch] {time.strftime('%H:%M:%S')} {msg}", flush=True)
        if proc.returncode == 0:
            print("[watch] all layers complete; running full check", flush=True)
            run = [py, str(ROOT / "scripts" / "full_check.py"), "--model-dir", args.model_dir]
            if args.bench:
                run.append("--bench")
                run.extend(["--decode", "16"])
            subprocess.run(run)
            if args.bench:
                # end-to-end generation smoke on the real model
                print("[watch] running generation smoke", flush=True)
                gen_script = (
                    "import sys; sys.path.insert(0, 'src'); "
                    "from tinyinfer.engine import Engine; from tinyinfer.tiers import OptLevel; "
                    "from transformers import AutoTokenizer; "
                    "d = '" + args.model_dir + "'; "
                    "tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True); "
                    "eng = Engine(d, tier=OptLevel.BALANCED, device='cuda'); "
                    "prompt = tok('Hello, I am DeepSeek.', return_tensors='pt').input_ids[0].tolist(); "
                    "out = eng.generate(prompt, max_new_tokens=16, temperature=0.0); "
                    "print('generated:', repr(tok.decode(out)))"
                )
                subprocess.run([py, "-c", gen_script])
            return 0
        time.sleep(args.interval)
    print("[watch] timed out waiting for the download", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
