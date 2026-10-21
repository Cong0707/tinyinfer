"""Call the local server through its OpenAI-compatible endpoint.

    python -m tinyinfer.serve --model <dir> --port 8000 &
    PYTHONPATH=src python examples/openai_client.py --prompt "Once upon a time"

Uses only the standard library so it runs on a bare Python install.
"""
import argparse
import json
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    body = json.dumps({"prompt": args.prompt, "max_tokens": args.max_tokens}).encode("utf-8")
    req = urllib.request.Request(args.url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    text = choice.get("text") or choice.get("message", {}).get("content", "")
    print(text)
    print(json.dumps(data.get("usage", {}), indent=2))


if __name__ == "__main__":
    main()
