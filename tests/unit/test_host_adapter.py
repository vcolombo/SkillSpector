# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from skillspector.providers.host.adapter import PluginLlmChatModel


class _Schema(BaseModel):
    ok: bool
    note: str


class _Result:
    def __init__(self, *, text: str = "", parsed: dict | None = None) -> None:
        self.text = text
        self.parsed = parsed


class _FakeHostLlm:
    """Records calls and enforces the real ``agent.plugin_llm.PluginLlm`` contract
    (verified against Hermes 0.17.0): structured calls require non-empty
    ``instructions`` AND at least one input block; parsed data is ``result.parsed``.
    """

    def __init__(self, *, text: str = "hello", parsed: dict | None = None) -> None:
        self._text = text
        self._parsed = parsed
        self.calls: list[dict] = []

    async def acomplete(self, messages, *, purpose=None, **kw):
        self.calls.append({"kind": "text", "messages": messages, "purpose": purpose, **kw})
        return _Result(text=self._text)

    async def acomplete_structured(self, *, instructions, input, json_schema, purpose=None, **kw):
        if not instructions or not instructions.strip():
            raise ValueError("acomplete_structured requires non-empty instructions")
        if not input:
            raise ValueError("acomplete_structured requires at least one input block")
        self.calls.append(
            {
                "kind": "structured",
                "instructions": instructions,
                "input": list(input),
                "schema": json_schema,
                "purpose": purpose,
                **kw,
            }
        )
        return _Result(text=self._text, parsed=self._parsed)


def test_text_path_calls_acomplete_and_returns_message() -> None:
    host = _FakeHostLlm(text="analysis text")
    model = PluginLlmChatModel(host)
    msg = asyncio.run(model.ainvoke("prompt body"))
    assert msg.text == "analysis text"
    assert host.calls[0]["kind"] == "text"
    assert host.calls[0]["purpose"] == "skillspector-scan"
    assert host.calls[0]["messages"] == [{"role": "user", "content": "prompt body"}]


def test_structured_path_validates_into_schema() -> None:
    host = _FakeHostLlm(parsed={"ok": True, "note": "clean"})
    structured = PluginLlmChatModel(host).with_structured_output(_Schema)
    result = asyncio.run(structured.ainvoke("prompt"))
    assert isinstance(result, _Schema)
    assert result.ok is True and result.note == "clean"
    assert host.calls[0]["kind"] == "structured"
    assert host.calls[0]["schema"] == _Schema.model_json_schema()
    # The analyzer prompt travels as the (required) input block.
    assert host.calls[0]["input"] == [{"type": "text", "text": "prompt"}]


def test_structured_path_falls_back_to_text_json() -> None:
    host = _FakeHostLlm(text='{"ok": false, "note": "from text"}')
    structured = PluginLlmChatModel(host).with_structured_output(_Schema)
    result = asyncio.run(structured.ainvoke("prompt"))
    assert result.ok is False and result.note == "from text"


def test_structured_path_raises_clear_error_on_unparseable_response() -> None:
    # Neither a parsed object (parsed=None) nor valid JSON text → a clear error,
    # not a cryptic json.loads failure on a repr string.
    host = _FakeHostLlm(text="not valid json {{{")
    structured = PluginLlmChatModel(host).with_structured_output(_Schema)
    with pytest.raises(ValueError, match="neither a parsed object"):
        asyncio.run(structured.ainvoke("prompt"))


class _PluginLlmTrustError(Exception):
    """Stand-in matched by name — the adapter checks type(exc).__name__."""


# The adapter matches the exception by type name; give the stand-in the real name.
_PluginLlmTrustError.__name__ = "PluginLlmTrustError"


class _GatedHostLlm(_FakeHostLlm):
    """Rejects any model override like an un-opted-in host trust gate."""

    async def acomplete_structured(self, *, instructions, input, json_schema, purpose=None, **kw):
        if kw.get("model"):
            raise _PluginLlmTrustError("model override not allowed")
        return await super().acomplete_structured(
            instructions=instructions, input=input, json_schema=json_schema, purpose=purpose
        )


def test_trust_gate_rejection_retries_without_override() -> None:
    # Operator set SKILLSPECTOR_MODEL but the host config doesn't allow model
    # overrides: degrade to the host default model, don't fail the analyzer.
    host = _GatedHostLlm(parsed={"ok": True, "note": "retried"})
    structured = PluginLlmChatModel(host, model="operator-model").with_structured_output(_Schema)
    result = asyncio.run(structured.ainvoke("prompt"))
    assert result.note == "retried"
    # Only the successful (override-free) call reached the recording fake.
    assert len(host.calls) == 1 and "model" not in host.calls[0]


def test_non_trust_errors_are_not_retried() -> None:
    class _BrokenHostLlm(_FakeHostLlm):
        async def acomplete_structured(self, **kw):
            raise RuntimeError("provider blew up")

    structured = PluginLlmChatModel(
        _BrokenHostLlm(), model="operator-model"
    ).with_structured_output(_Schema)
    with pytest.raises(RuntimeError, match="provider blew up"):
        asyncio.run(structured.ainvoke("prompt"))


def test_batch_and_stream_fail_loudly() -> None:
    model = PluginLlmChatModel(_FakeHostLlm())
    with pytest.raises(NotImplementedError):
        model.batch("x")
    with pytest.raises(NotImplementedError):
        model.stream("x")
