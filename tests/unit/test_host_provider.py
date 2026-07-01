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

from skillspector.providers.host import reset_host_llm, set_host_llm
from skillspector.providers.host.adapter import PluginLlmChatModel
from skillspector.providers.host.provider import HostLLMProvider


class _FakeHostLlm:
    async def acomplete(self, **kw):  # pragma: no cover - not called here
        raise AssertionError

    async def acomplete_structured(self, **kw):  # pragma: no cover
        raise AssertionError


class _RecordingHostLlm:
    """Records the kwargs of each ``acomplete`` call (to observe model forwarding)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def acomplete(self, **kw):
        self.calls.append(kw)

        class _Result:
            text = "ok"

        return _Result()


def test_available_only_when_host_llm_bound() -> None:
    provider = HostLLMProvider()
    available, reason = provider.is_available()
    assert available is False and reason

    token = set_host_llm(_FakeHostLlm())
    try:
        available, reason = provider.is_available()
        assert available is True and reason is None
    finally:
        reset_host_llm(token)


def test_no_credentials_and_unknown_metadata() -> None:
    provider = HostLLMProvider()
    assert provider.resolve_credentials() is None
    assert provider.get_context_length("host") is None
    assert provider.get_max_output_tokens("host") is None
    assert provider.resolve_model() == "host"


def test_create_chat_model_returns_adapter_bound_to_host() -> None:
    host = _FakeHostLlm()
    token = set_host_llm(host)
    try:
        model = HostLLMProvider().create_chat_model("host", max_tokens=1024)
        assert isinstance(model, PluginLlmChatModel)
    finally:
        reset_host_llm(token)


def test_create_chat_model_none_when_unbound() -> None:
    assert HostLLMProvider().create_chat_model("host", max_tokens=1024) is None


def test_create_chat_model_forwards_only_non_sentinel_model() -> None:
    # The "host" sentinel means "use whatever the host is using" — not
    # forwarded. Any other label (operator override or explicit caller
    # model=) reaches ctx.llm as a model kwarg.
    host = _RecordingHostLlm()
    token = set_host_llm(host)
    try:
        sentinel = HostLLMProvider().create_chat_model("host", max_tokens=1024)
        asyncio.run(sentinel.ainvoke("p"))
        assert "model" not in host.calls[0]

        explicit = HostLLMProvider().create_chat_model("some-model", max_tokens=1024)
        asyncio.run(explicit.ainvoke("p"))
        assert host.calls[1]["model"] == "some-model"
    finally:
        reset_host_llm(token)
