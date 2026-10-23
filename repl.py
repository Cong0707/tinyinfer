"""Interactive chat against the resident tinyinfer server (OpenAI-compatible API).

    python repl.py                 # interactive; Ctrl+C / Ctrl+D to quit
    python repl.py "问题"           # one-shot question

Streams the answer with the server's SSE endpoint, the same surface any OpenAI client
would use -- there is no private path into the engine.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash-1017"
# One conversation = one session id = one .kvcache/<id>.pt. The mapping lives in a small
# sqlite file so asking the same one-shot question again resumes the same KV instead of
# re-prefilling it, and so different questions never share a KV file.
DB = Path(__file__).resolve().parent / ".kvcache" / "sessions.db"


def _db():
    DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "key TEXT PRIMARY KEY, session_id TEXT, turns INTEGER DEFAULT 0, updated REAL)"
    )
    return conn


def session_for(key: str) -> str:
    conn = _db()
    try:
        row = conn.execute("SELECT session_id FROM sessions WHERE key=?", (key,)).fetchone()
        if row:
            conn.execute("UPDATE sessions SET turns=turns+1, updated=? WHERE key=?",
                         (time.time(), key))
            conn.commit()
            return row[0]
        sid = uuid.uuid4().hex[:12]
        conn.execute("INSERT INTO sessions (key, session_id, turns, updated) VALUES (?,?,1,?)",
                     (key, sid, time.time()))
        conn.commit()
        return sid
    finally:
        conn.close()



def _post(messages, max_tokens: int, temperature: float, session_id: str | None = None):
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
         "temperature": temperature, "stream": True,
         "stream_options": {"include_usage": True}}
    ).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json",
                 "X-Session-Id": session_id or "repl"},
    )
    return urllib.request.urlopen(req, timeout=900)


def chat(messages, max_tokens: int = 64, temperature: float = 1.0,
         session_id: str | None = None) -> str:
    """Stream one completion to stdout, print the counters, return the text."""
    started = time.perf_counter()
    first = None
    chunks = 0
    usage = None
    parts: list[str] = []
    try:
        with _post(messages, max_tokens, temperature, session_id) as resp:
            sys.stdout.write("model: ")
            sys.stdout.flush()
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if not piece:
                        continue
                    if first is None:
                        first = time.perf_counter() - started
                    chunks += 1
                    parts.append(piece)
                    sys.stdout.write(piece)
                    sys.stdout.flush()
    except urllib.error.URLError as exc:
        print(f"\n[error] {exc} (is the server running? start.ps1 -Bg)")
        return ""
    elapsed = time.perf_counter() - started
    first = first or 0.0
    decode = max(elapsed - first, 1e-6)
    rate = chunks / decode
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion = (usage or {}).get("completion_tokens", chunks)
    up = prompt_tokens if prompt_tokens is not None else "?"
    # first = time to the first streamed token (prefill + queueing); the rest is decode
    print(f"  ({rate:.1f} tok/s  \u2191{up} \u2193{completion}  "
          f"{first:.0f}+{max(elapsed - first, 0.0):.0f}s)")
    return "".join(parts)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    one_shot = " ".join(sys.argv[1:]).strip()
    messages: list[dict[str, str]] = []
    # interactive mode: the session is keyed by the FIRST user message, so quitting and
    # restarting the REPL, then asking the same opening question, resumes that
    # conversation's KV instead of re-prefilling it
    session: str | None = None
    if one_shot:
        # same question -> same session -> the server resumes its KV from .kvcache
        sid = session_for(hashlib.sha1(one_shot.encode("utf-8")).hexdigest())
        chat([{"role": "user", "content": one_shot}], session_id=sid)
        return 0
    while True:
        try:
            user = input("user: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        user = user.strip()
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        if session is None:
            session = session_for("interactive:" + hashlib.sha1(user.encode("utf-8")).hexdigest())
        messages.append({"role": "user", "content": user})
        reply = chat(messages, session_id=session)
        if reply:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())