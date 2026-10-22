# Prompt layout tests for the checkpoint's turn encoder (tinyinfer.chat).
import pytest

from tinyinfer.chat import (
    ASSISTANT,
    BOS,
    EOS,
    SYSTEM,
    THINKING_END,
    THINKING_START,
    USER,
    encode_messages,
)


def test_single_turn_chat_mode():
    prompt = encode_messages([{"role": "user", "content": "hi"}])
    assert prompt == BOS + USER + "hi" + ASSISTANT + THINKING_END


def test_multi_turn_with_system():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    prompt = encode_messages(messages)
    assert prompt == (
        BOS
        + SYSTEM + "be brief"
        + USER + "hi"
        + ASSISTANT + THINKING_END + "hello" + EOS
        + USER + "again"
        + ASSISTANT + THINKING_END
    )


def test_thinking_mode_opens_reasoning():
    prompt = encode_messages([{"role": "user", "content": "hi"}], thinking_mode="thinking")
    assert prompt.endswith(ASSISTANT + THINKING_START)
    # the numeric budget rides in a leading system message, like the reference encoder
    assert "Reasoning Effort: 75 " in prompt
    assert prompt.startswith(BOS + SYSTEM + "Reasoning Effort: 75")


def test_historical_reasoning_is_dropped():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "secret"},
        {"role": "user", "content": "q2"},
    ]
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert "secret" not in prompt
    assert prompt.endswith(ASSISTANT + THINKING_START)
    keep = encode_messages(messages, thinking_mode="thinking", drop_thinking=False)
    assert "secret" in keep


def test_current_turn_reasoning_is_kept():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "why"},
    ]
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert "why" + THINKING_END + "a1" in prompt


def test_rejects_unknown_mode_or_role():
    with pytest.raises(ValueError):
        encode_messages([{"role": "user", "content": "hi"}], thinking_mode="verbose")
    with pytest.raises(ValueError):
        encode_messages([{"role": "tool", "content": "x"}])


def test_content_parts_are_flattened():
    prompt = encode_messages(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    )
    assert prompt == BOS + USER + "ab" + ASSISTANT + THINKING_END

def test_server_renders_the_checkpoint_turn_format():
    # the shipped tokenizer has no chat_template, so the backend must fall back to the
    # checkpoint encoder instead of a generic "role: content" rendering
    from tinyinfer.server import GenerationRequest, TinyModelBackend

    class _Tok:
        chat_template = None

    backend = TinyModelBackend.__new__(TinyModelBackend)
    backend._tokenizer = _Tok()
    request = GenerationRequest(
        request_id="t", messages=[{"role": "user", "content": "hi"}]
    )
    assert backend.render_prompt(request) == BOS + USER + "hi" + ASSISTANT + THINKING_END
