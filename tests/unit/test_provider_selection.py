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

import pytest

from skillspector.providers import _select_active_provider
from skillspector.providers.host import reset_host_llm, set_host_llm
from skillspector.providers.host.provider import HostLLMProvider
from skillspector.providers.nv_build import NvBuildProvider


class _FakeHostLlm:
    pass


def test_bound_host_llm_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai")
    token = set_host_llm(_FakeHostLlm())
    try:
        assert isinstance(_select_active_provider(), HostLLMProvider)
    finally:
        reset_host_llm(token)


def test_auto_env_selects_host_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "auto")
    assert isinstance(_select_active_provider(), HostLLMProvider)


def test_unset_still_defaults_to_nv_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKILLSPECTOR_PROVIDER", raising=False)
    assert isinstance(_select_active_provider(), NvBuildProvider)
