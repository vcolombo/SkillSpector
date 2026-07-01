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

"""``HostLLMProvider`` — delegates the LLM pass to the Hermes host ``ctx.llm``.

Availability is capability-based (is the host LLM bound in this context?) rather
than credential-based: the host owns keys, provider resolution, and audit. Model
metadata is unknown here (the host chooses the model), so the metadata methods
return ``None`` and callers fall back to defaults.
"""

from __future__ import annotations

import os

from skillspector.providers.host._state import get_host_llm
from skillspector.providers.host.adapter import PluginLlmChatModel


class HostLLMProvider:
    DEFAULT_MODEL = "host"
    SLOT_DEFAULTS: dict[str, str] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        return None

    def is_available(self) -> tuple[bool, str | None]:
        if get_host_llm() is None:
            return False, "No host LLM is bound (not running inside a Hermes plugin call)."
        return True, None

    def get_context_length(self, model: str) -> int | None:
        return None

    def get_max_output_tokens(self, model: str) -> int | None:
        return None

    def resolve_model(self, slot: str = "default") -> str:
        # The host owns model selection; honour an explicit operator override if
        # present, else the slot default, else a non-empty sentinel (required by
        # the provider protocol). Mirrors the standard provider waterfall
        # (env → slot default → general default), even though SLOT_DEFAULTS is
        # empty today — so populating it later behaves like every other provider.
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> PluginLlmChatModel | None:
        host_llm = get_host_llm()
        if host_llm is None:
            return None
        # Forward ONLY an explicit operator override (SKILLSPECTOR_MODEL) to
        # ctx.llm — never the `model` parameter. Callers pass SkillSpector's
        # own model labels (analyzer defaults like the nv_build names), which
        # are meaningless in the host's model namespace; forwarding one trips
        # the host's trust gate (PluginLlmTrustError: model override not
        # allowed) and kills that analyzer's call — observed live on Hermes
        # 0.17.0, where semantic_security_discovery passes an explicit label
        # while the other analyzers use the "host" sentinel. The parameter is
        # still honoured for token budgeting by the caller; host model choice
        # belongs to the host unless the operator says otherwise.
        override_model = os.environ.get("SKILLSPECTOR_MODEL", "").strip() or None
        return PluginLlmChatModel(host_llm, model=override_model)
