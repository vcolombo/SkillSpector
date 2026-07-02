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

"""Tests for routing ``get_chat_model`` / ``is_llm_available`` to the host provider.

When a host LLM is bound (see ``skillspector.providers.host``), the active
provider resolves to ``HostLLMProvider`` and both entry points must treat it
like a CLI provider: capability-based availability, and a chat model returned
without going through the credential-resolution path.
"""

from __future__ import annotations

from skillspector.llm_utils import get_chat_model, is_llm_available
from skillspector.providers.host import reset_host_llm, set_host_llm
from skillspector.providers.host.adapter import PluginLlmChatModel


class _FakeHostLlm:
    async def acomplete(self, **kw):  # pragma: no cover
        raise AssertionError

    async def acomplete_structured(self, **kw):  # pragma: no cover
        raise AssertionError


def test_is_llm_available_true_when_host_bound() -> None:
    token = set_host_llm(_FakeHostLlm())
    try:
        available, reason = is_llm_available()
        assert available is True and reason is None
    finally:
        reset_host_llm(token)


def test_get_chat_model_returns_host_adapter() -> None:
    token = set_host_llm(_FakeHostLlm())
    try:
        assert isinstance(get_chat_model(), PluginLlmChatModel)
    finally:
        reset_host_llm(token)
