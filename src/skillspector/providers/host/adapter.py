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

"""LangChain-compatible adapter over the Hermes host LLM (``ctx.llm``).

Implements only the surface the LLM analyzers use — ``invoke`` / ``ainvoke``
/ ``with_structured_output`` — backed by the host's async ``acomplete`` and
``acomplete_structured``. ``batch`` / ``stream`` are stubbed to fail loudly,
mirroring :class:`skillspector.llm_utils.AgentCLIChatModel`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, NoReturn

from langchain_core.messages import AIMessage

_PURPOSE = "skillspector-scan"

# The host's structured API takes a short system-level directive in
# ``instructions`` and the actual task content as input blocks (both are
# required to be non-empty — verified against agent.plugin_llm in Hermes
# 0.17.0). The analyzer prompt therefore travels as the input block.
_STRUCTURED_INSTRUCTIONS = (
    "Complete the analysis task described in the input. Respond with a single "
    "JSON object conforming to the provided JSON schema."
)


def _result_text(result: Any) -> str:
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _override_kwargs(provider: str | None, model: str | None) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model
    return kwargs


def _is_trust_error(exc: Exception) -> bool:
    """True when *exc* is the host's override trust-gate rejection.

    Matched by type name (``PluginLlmTrustError``) because SkillSpector never
    imports the Hermes ``agent`` package.
    """
    return type(exc).__name__ == "PluginLlmTrustError"


async def _call_with_override_fallback(call: Any, kwargs: dict[str, Any]) -> Any:
    """Invoke *call* with override kwargs; on a trust-gate rejection retry once
    without them.

    An operator may set SKILLSPECTOR_MODEL without enabling
    ``allow_model_override`` in the host config — degrade to the host's default
    model rather than failing the analyzer call.
    """
    overrides = {k: kwargs.pop(k) for k in ("provider", "model") if k in kwargs}
    if not overrides:
        return await call(**kwargs)
    try:
        return await call(**kwargs, **overrides)
    except Exception as exc:  # noqa: BLE001 — retry only the trust rejection
        if not _is_trust_error(exc):
            raise
        return await call(**kwargs)


def _run_sync(coro: Any) -> Any:
    """Run *coro* to completion from a synchronous caller.

    The Hermes scan path is async (``arun_batches`` → ``ainvoke``); ``invoke``
    exists only for the synchronous ``run_batches`` / ``chat_completion`` paths,
    which are not used under Hermes. Fails clearly if called inside a running loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()  # avoid a "coroutine was never awaited" warning
    raise RuntimeError(
        "PluginLlmChatModel.invoke() cannot be used inside a running event loop; "
        "use `await ainvoke(...)` instead."
    )


class _StructuredPluginLlmModel:
    def __init__(
        self, host_llm: object, schema: type, *, provider: str | None, model: str | None
    ) -> None:
        self._host = host_llm
        self._schema = schema
        self._provider = provider
        self._model = model

    async def ainvoke(self, prompt: str) -> Any:
        result = await _call_with_override_fallback(
            self._host.acomplete_structured,  # type: ignore[attr-defined]
            {
                "instructions": _STRUCTURED_INSTRUCTIONS,
                "input": [{"type": "text", "text": prompt}],
                "json_schema": self._schema.model_json_schema(),
                "purpose": _PURPOSE,
                **_override_kwargs(self._provider, self._model),
            },
        )
        # PluginLlmStructuredResult.parsed is set only when the response was
        # valid JSON; otherwise fall back to extracting JSON from the text.
        data = getattr(result, "parsed", None)
        if not isinstance(data, dict):
            raw = _result_text(result)
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    "Host LLM structured response was neither a parsed object "
                    f"(result.parsed) nor valid JSON text: {raw[:200]!r}"
                ) from exc
        return self._schema.model_validate(data)

    def invoke(self, prompt: str) -> Any:
        return _run_sync(self.ainvoke(prompt))


class PluginLlmChatModel:
    def __init__(
        self, host_llm: object, *, provider: str | None = None, model: str | None = None
    ) -> None:
        self._host = host_llm
        self._provider = provider
        self._model = model

    def batch(self, *args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError(
            "PluginLlmChatModel supports only invoke/ainvoke/with_structured_output; "
            "batch() is not available for the host LLM."
        )

    def stream(self, *args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError(
            "PluginLlmChatModel supports only invoke/ainvoke/with_structured_output; "
            "stream() is not available for the host LLM."
        )

    async def ainvoke(self, prompt: str) -> AIMessage:
        result = await _call_with_override_fallback(
            self._host.acomplete,  # type: ignore[attr-defined]
            {
                "messages": [{"role": "user", "content": prompt}],
                "purpose": _PURPOSE,
                **_override_kwargs(self._provider, self._model),
            },
        )
        return AIMessage(content=_result_text(result))

    def invoke(self, prompt: str) -> AIMessage:
        return _run_sync(self.ainvoke(prompt))

    def with_structured_output(self, schema: type) -> _StructuredPluginLlmModel:
        return _StructuredPluginLlmModel(
            self._host, schema, provider=self._provider, model=self._model
        )
