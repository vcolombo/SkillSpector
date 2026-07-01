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

"""End-to-end lock-in for the host-LLM routing.

Unlike the unit tests (which stop at the adapter/provider/selection plumbing),
this drives the *real* ``run_scan`` graph with ``use_llm=True`` and only a bound
fake host LLM — no monkeypatching of ``get_chat_model``. It proves the whole
chain works: the ContextVar set at the plugin boundary propagates into the
LangGraph analyzer nodes, ``is_llm_available()`` reports the host available,
and the LLM pass actually routes through :class:`PluginLlmChatModel` to the
host's ``acomplete_structured``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skillspector.mcp_server import run_scan
from skillspector.providers.host import reset_host_llm, set_host_llm

# Credential env vars that could otherwise make the LLM pass "available" for a
# provider even with no host bound — cleared in the negative test so its result
# is deterministic regardless of the runner's environment.
_CREDENTIAL_ENV = (
    "SKILLSPECTOR_PROVIDER",
    "NVIDIA_INFERENCE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_PROXY_API_KEY",
)

_SAFE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "safe_skill"


class _Result:
    """Stand-in for a ``PluginLlm`` result: parsed data on ``.parsed``."""

    def __init__(self, *, text: str = "", parsed: dict | None = None) -> None:
        self.text = text
        self.parsed = parsed


class _FakeHostLlm:
    """Duck-typed host LLM that records invocations and returns clean results.

    Enforces the real ``PluginLlm`` contract (verified against Hermes 0.17.0):
    structured calls need non-empty ``instructions`` and at least one input
    block. ``{"findings": []}`` validates against both structured schemas the
    analyzers request (``LLMAnalysisResult`` and ``MetaAnalyzerResult`` — each
    has ``findings: list = Field(default_factory=list)``), so a real scan
    completes cleanly while proving the host was actually driven.
    """

    def __init__(self) -> None:
        self.structured_calls = 0
        self.text_calls = 0

    async def acomplete(self, messages, *, purpose=None, **kwargs):
        self.text_calls += 1
        return _Result(text="")

    async def acomplete_structured(
        self, *, instructions, input, json_schema, purpose=None, **kwargs
    ):
        if not instructions or not instructions.strip():
            raise ValueError("acomplete_structured requires non-empty instructions")
        if not input:
            raise ValueError("acomplete_structured requires at least one input block")
        self.structured_calls += 1
        return _Result(parsed={"findings": []})


def test_run_scan_routes_llm_pass_through_bound_host() -> None:
    host = _FakeHostLlm()
    token = set_host_llm(host)
    try:
        verdict = asyncio.run(run_scan(str(_SAFE_FIXTURE), use_llm=True))
    finally:
        reset_host_llm(token)

    # The gate saw the host as available and the semantic pass ran...
    assert verdict["llm_available"] is True
    assert verdict["llm_used"] is True
    assert verdict["scan_mode"] == "static+llm"
    # ...and it actually routed through the host adapter (not a no-op).
    assert host.structured_calls > 0
    # The clean fixture with empty host findings is still a clean verdict.
    assert verdict["safe_to_install"] is True


def test_run_scan_without_bound_host_is_static_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # No host bound and no provider credentials → the LLM pass is correctly
    # reported as not run, even though it was requested. Clearing the credential
    # env makes availability deterministically False on any runner.
    for name in _CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)

    verdict = asyncio.run(run_scan(str(_SAFE_FIXTURE), use_llm=True))

    assert verdict["llm_requested"] is True
    assert verdict["llm_available"] is False
    assert verdict["llm_used"] is False
    assert verdict["scan_mode"] == "static-only"
