# Official DeepSeek-V4.1 prompt encoding (text-only turns, no tools).
#
# The checkpoint's tokenizer ships without a chat template, so apply_chat_template()
# fails and a generic role: content rendering sends the model a format it was never
# trained on. The release carries the real encoder in encoding/encoding.py; this module
# mirrors its text-only path for the turns the server accepts (system, user, assistant)
# in chat and thinking mode.
#
# Layout (chat mode, the default):
#
#     <bos><|User|>question<|Assistant|></think>answer<|end of sentence|>
#
# The header token after <|Assistant|> is </think> in chat mode (the answer follows
# directly) and <think> in thinking mode; a turn whose reasoning was dropped starts
# with </think> as well. Thinking mode also carries a numeric reasoning-effort budget
# in a leading <|System|> message, exactly like the reference encoder.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
SYSTEM = "<\uff5cSystem\uff5c>"
USER = "<\uff5cUser\uff5c>"
ASSISTANT = "<\uff5cAssistant\uff5c>"
LATEST_REMINDER = "<\uff5clatest_reminder\uff5c>"
THINKING_START = "<think>"
THINKING_END = "</think>"

VALID_MODES = ("chat", "thinking")

# numeric reasoning-effort budget, thinking mode only (reference: DEFAULT_REASONING_EFFORT)
REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
DEFAULT_REASONING_EFFORT = "high"


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return "" if content is None else str(content)


def _last_user_index(messages: Sequence[dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ("user", "system"):
            return idx
    return -1


def _effort_prefix(thinking_mode: str, reasoning_effort: Any) -> str:
    # index 0 only: the budget rides in a leading system message and exists only in
    # thinking mode, so chat mode is byte-identical to the plain layout above
    if thinking_mode != "thinking":
        return ""
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_REASONING_EFFORT
    if isinstance(reasoning_effort, str):
        if reasoning_effort not in REASONING_EFFORT_MAPPINGS:
            raise ValueError(
                f"reasoning_effort must be an int in [1, 100] or one of "
                f"{sorted(REASONING_EFFORT_MAPPINGS)}, got {reasoning_effort!r}"
            )
        effort = REASONING_EFFORT_MAPPINGS[reasoning_effort]
    else:
        effort = int(reasoning_effort)
        if not 1 <= effort <= 100:
            raise ValueError(f"reasoning_effort must be an int in [1, 100], got {effort}")
    return SYSTEM + REASONING_EFFORT_TEMPLATE.format(budget=effort)


def encode_messages(
    messages: Sequence[dict[str, Any]],
    thinking_mode: str = "chat",
    add_bos: bool = True,
    drop_thinking: bool = True,
    reasoning_effort: Any = None,
) -> str:
    # Render OpenAI-style messages into the checkpoint's prompt format.
    if thinking_mode not in VALID_MODES:
        raise ValueError(f"thinking_mode must be one of {VALID_MODES}, got {thinking_mode!r}")
    messages = list(messages)
    if not messages:
        return BOS if add_bos else ""
    last_user = _last_user_index(messages)
    prompt = BOS if add_bos else ""
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = _text(msg.get("content"))
        prefix = _effort_prefix(thinking_mode, reasoning_effort) if idx == 0 else ""
        if role == "system":
            prompt += prefix + SYSTEM + content
        elif role == "user":
            prompt += prefix + USER + content
        elif role == "assistant":
            reasoning = _text(msg.get("reasoning_content"))
            # historical reasoning is dropped before the last user turn, exactly like
            # the reference encoder; the current turn keeps it verbatim
            if thinking_mode == "thinking" and (not drop_thinking or idx > last_user):
                prompt += prefix + reasoning + THINKING_END
            else:
                prompt += prefix
            prompt += content + EOS
        elif role in ("latest_reminder", "reminder"):
            prompt += prefix + LATEST_REMINDER + content
        else:
            raise ValueError(f"unsupported role {role!r}")
        nxt = messages[idx + 1].get("role") if idx + 1 < len(messages) else None
        if nxt not in (None, "assistant", "latest_reminder"):
            continue
        if role in ("user", "system"):
            prompt += ASSISTANT
            keep_thinking = thinking_mode == "thinking" and (
                not drop_thinking or idx >= last_user
            )
            prompt += THINKING_START if keep_thinking else THINKING_END
    return prompt
