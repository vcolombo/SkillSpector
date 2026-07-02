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

"""Process-wide injection point for the Hermes host LLM.

The Hermes plugin sets the host ``ctx.llm`` here for the duration of a scan;
:func:`skillspector.providers._select_active_provider` reads it to route the
LLM pass through the host instead of a credentialed provider. A ``ContextVar``
(not a plain global) keeps the value task-local and async-safe, and avoids the
process-``env`` mutation the old plugin relied on.

The value is an opaque ``object`` — SkillSpector never imports the Hermes
``agent`` package. The adapter calls its methods by duck typing.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_HOST_LLM: ContextVar[object | None] = ContextVar("skillspector_host_llm", default=None)


def set_host_llm(host_llm: object) -> Token:
    """Bind *host_llm* for the current context; returns a reset token."""
    return _HOST_LLM.set(host_llm)


def get_host_llm() -> object | None:
    """Return the host LLM bound in the current context, or ``None``."""
    return _HOST_LLM.get()


def reset_host_llm(token: Token) -> None:
    """Undo a prior :func:`set_host_llm`, restoring the previous binding."""
    _HOST_LLM.reset(token)
