"""Chat prompt rendering: the two formats the checkpoint actually accepts.

The checkpoint ships without a chat_template, so server.render_prompt falls back to
tinyinfer.chat.encode_messages. That fallback decides what the model sees -- getting the
`</think>` vs `<think>` terminator backwards turns a direct answer into a chain of
thought, which looks like a quality regression rather than a formatting bug.
"""
from __future__ import annotations

from . import Case


def register(report) -> None:
    try:
        from ..chat import encode_messages
    except Exception as exc:
        report.add(Case('chat_render', f'unavailable: {exc!r}', False))
        return

    msgs = [{'role': 'user', 'content': 'hi'}]
    try:
        chat = encode_messages(msgs, thinking_mode='chat')
        think = encode_messages(msgs, thinking_mode='thinking')
    except Exception as exc:
        report.add(Case('chat_render', f'raised {exc!r}', False))
        return

    # direct-answer mode closes the thinking block immediately
    report.add(Case('chat mode ends with a closed thinking block',
                    repr(chat[-24:]), chat.rstrip().endswith('</think>')))
    report.add(Case('chat mode opens no thinking block',
                    '', '<think>' not in chat))
    # reasoning mode opens one and leaves it to be filled
    report.add(Case('thinking mode ends with an open thinking block',
                    repr(think[-20:]), think.rstrip().endswith('<think>')))
    report.add(Case('thinking mode asks for reasoning effort',
                    '', 'Reasoning Effort' in think))
    # a system message must survive in both
    with_sys = [{'role': 'system', 'content': 'be brief'}] + msgs
    try:
        s_chat = encode_messages(with_sys, thinking_mode='chat')
        s_think = encode_messages(with_sys, thinking_mode='thinking')
        report.add(Case('a user system message is carried in chat mode',
                        '', 'be brief' in s_chat))
        report.add(Case('a user system message is carried in thinking mode',
                        '', 'be brief' in s_think))
    except Exception as exc:
        report.add(Case('chat_render with system', f'raised {exc!r}', False))
