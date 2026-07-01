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

"""Tests for the capability-aware LLM availability gate in ``run_scan``."""

from __future__ import annotations

import asyncio

import pytest

from skillspector import mcp_server


class _FakeGraph:
    async def ainvoke(self, state, config=None):
        return {"findings": [], "filtered_findings": []}


def test_gate_uses_is_llm_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "graph", _FakeGraph())

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (True, None))
    verdict = asyncio.run(mcp_server.run_scan("some/target", use_llm=True))
    assert verdict["llm_available"] is True
    assert verdict["llm_used"] is True

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    verdict = asyncio.run(mcp_server.run_scan("some/target", use_llm=True))
    assert verdict["llm_available"] is False
    assert verdict["llm_used"] is False
