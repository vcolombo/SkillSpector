# Hermes Host-LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let SkillSpector's optional LLM semantic pass run against the Hermes host's already-configured model (`ctx.llm`) when invoked as a plugin, with zero credentials managed by the plugin, while leaving standalone CLI/MCP behavior unchanged.

**Architecture:** Add a credential-less `HostLLMProvider` whose LangChain-compatible adapter (`PluginLlmChatModel`) forwards to the host `ctx.llm` (`acomplete`/`acomplete_structured`). The host LLM is injected via a `ContextVar` set at the plugin boundary and read in `_select_active_provider()`; no process-env mutation. The `run_scan` availability gate switches from credential-check to the capability-aware `is_llm_available()`.

**Tech Stack:** Python 3, LangChain-core, Pydantic v2, pytest, ruff. Hermes plugin API (`ctx.register_tool`, `ctx.llm`).

## Global Constraints

- Every new `.py` file starts with the SPDX Apache-2.0 header block copied verbatim from any existing `src/skillspector/*.py` file (13 lines, `# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES...`).
- ruff line-length **100**; run `ruff check` and `ruff format --check` clean before every commit.
- Tests run under `pytest -m "not integration"`; new unit tests live in `tests/unit/`.
- **No AI attribution** in commit messages or code comments.
- **Do NOT import the Hermes `agent` package at runtime** anywhere in `src/skillspector/` — SkillSpector must stay installable standalone. The injected host LLM is held as a duck-typed `object`; type-only references go under `if TYPE_CHECKING:`.
- The injected host LLM contract this code assumes (must be validated in Task 1 against the real `agent.plugin_llm.PluginLlm`):
  - `await host_llm.acomplete(messages=[{"role": "user", "content": str}], purpose=str)` → result whose assistant text is `result.text`.
  - `await host_llm.acomplete_structured(instructions=str, input=[], json_schema=dict, purpose=str)` → result exposing parsed data as `result.output` (a dict) OR, failing that, JSON in `result.text`.
- Version stays `2.3.7` (repo current).

---

### Task 1: `PluginLlmChatModel` adapter

Bridges the host `ctx.llm` to the slice of the LangChain chat-model interface the analyzers use (`invoke`/`ainvoke`/`with_structured_output`). Mirrors the existing `AgentCLIChatModel` in `llm_utils.py`.

**Files:**
- Create: `src/skillspector/providers/host/__init__.py`
- Create: `src/skillspector/providers/host/_state.py`
- Create: `src/skillspector/providers/host/adapter.py`
- Test: `tests/unit/test_host_adapter.py`

**Interfaces:**
- Produces:
  - `PluginLlmChatModel(host_llm: object, *, provider: str | None = None, model: str | None = None)` with methods `invoke(prompt: str) -> AIMessage`, `async ainvoke(prompt: str) -> AIMessage`, `with_structured_output(schema: type) -> _StructuredPluginLlmModel`.
  - `_StructuredPluginLlmModel.invoke(prompt: str) -> BaseModel`, `async ainvoke(prompt: str) -> BaseModel`.
  - `_state.py`: `set_host_llm(host_llm: object) -> Token`, `get_host_llm() -> object | None`, `reset_host_llm(token: Token) -> None`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_host_adapter.py` (with the SPDX header):

```python
from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from skillspector.providers.host.adapter import PluginLlmChatModel


class _Schema(BaseModel):
    ok: bool
    note: str


class _Result:
    def __init__(self, *, text: str = "", output: dict | None = None) -> None:
        self.text = text
        self.output = output


class _FakeHostLlm:
    """Records calls and returns canned results matching the assumed contract."""

    def __init__(self, *, text: str = "hello", output: dict | None = None) -> None:
        self._text = text
        self._output = output
        self.calls: list[dict] = []

    async def acomplete(self, *, messages, purpose=None, **kw):
        self.calls.append({"kind": "text", "messages": messages, "purpose": purpose, **kw})
        return _Result(text=self._text)

    async def acomplete_structured(self, *, instructions, input, json_schema, purpose=None, **kw):
        self.calls.append(
            {"kind": "structured", "instructions": instructions, "schema": json_schema,
             "purpose": purpose, **kw}
        )
        return _Result(output=self._output or {})


def test_text_path_calls_acomplete_and_returns_message() -> None:
    host = _FakeHostLlm(text="analysis text")
    model = PluginLlmChatModel(host)
    msg = asyncio.run(model.ainvoke("prompt body"))
    assert msg.text == "analysis text"
    assert host.calls[0]["kind"] == "text"
    assert host.calls[0]["purpose"] == "skillspector-scan"
    assert host.calls[0]["messages"] == [{"role": "user", "content": "prompt body"}]


def test_structured_path_validates_into_schema() -> None:
    host = _FakeHostLlm(output={"ok": True, "note": "clean"})
    structured = PluginLlmChatModel(host).with_structured_output(_Schema)
    result = asyncio.run(structured.ainvoke("prompt"))
    assert isinstance(result, _Schema)
    assert result.ok is True and result.note == "clean"
    assert host.calls[0]["kind"] == "structured"
    assert host.calls[0]["schema"] == _Schema.model_json_schema()


def test_structured_path_falls_back_to_text_json() -> None:
    host = _FakeHostLlm(text='{"ok": false, "note": "from text"}')
    structured = PluginLlmChatModel(host).with_structured_output(_Schema)
    result = asyncio.run(structured.ainvoke("prompt"))
    assert result.ok is False and result.note == "from text"


def test_batch_and_stream_fail_loudly() -> None:
    model = PluginLlmChatModel(_FakeHostLlm())
    with pytest.raises(NotImplementedError):
        model.batch("x")
    with pytest.raises(NotImplementedError):
        model.stream("x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_host_adapter.py -q`
Expected: FAIL — `ModuleNotFoundError: skillspector.providers.host.adapter`.

- [ ] **Step 3: Write minimal implementation**

`src/skillspector/providers/host/_state.py` (with SPDX header):

```python
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
```

`src/skillspector/providers/host/adapter.py` (with SPDX header):

```python
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


def _run_sync(coro: Any) -> Any:
    """Run *coro* to completion from a synchronous caller.

    The Hermes scan path is async (``arun_batches`` → ``ainvoke``); ``invoke``
    exists only for the synchronous ``run_batches`` / ``chat_completion`` paths,
    which are not used under Hermes. Fails clearly if called inside a running loop.
    """
    return asyncio.run(coro)


class _StructuredPluginLlmModel:
    def __init__(
        self, host_llm: object, schema: type, *, provider: str | None, model: str | None
    ) -> None:
        self._host = host_llm
        self._schema = schema
        self._provider = provider
        self._model = model

    async def ainvoke(self, prompt: str) -> Any:
        result = await self._host.acomplete_structured(  # type: ignore[attr-defined]
            instructions=prompt,
            input=[],
            json_schema=self._schema.model_json_schema(),
            purpose=_PURPOSE,
            **_override_kwargs(self._provider, self._model),
        )
        data = getattr(result, "output", None)
        if not isinstance(data, dict):
            data = json.loads(_result_text(result))
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
        result = await self._host.acomplete(  # type: ignore[attr-defined]
            messages=[{"role": "user", "content": prompt}],
            purpose=_PURPOSE,
            **_override_kwargs(self._provider, self._model),
        )
        return AIMessage(content=_result_text(result))

    def invoke(self, prompt: str) -> AIMessage:
        return _run_sync(self.ainvoke(prompt))

    def with_structured_output(self, schema: type) -> _StructuredPluginLlmModel:
        return _StructuredPluginLlmModel(
            self._host, schema, provider=self._provider, model=self._model
        )
```

`src/skillspector/providers/host/__init__.py` (with SPDX header):

```python
"""Host-LLM provider package — routes SkillSpector's LLM pass through the
Hermes host ``ctx.llm`` (no plugin-managed credentials)."""

from __future__ import annotations

from ._state import get_host_llm, reset_host_llm, set_host_llm

__all__ = ["get_host_llm", "reset_host_llm", "set_host_llm"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_host_adapter.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Validate the host-LLM contract**

If the real `agent.plugin_llm.PluginLlm` differs from the assumed `acomplete`/`acomplete_structured` signatures or the `result.text`/`result.output` shape, adjust `adapter.py` and the `_FakeHostLlm` double together so the tests still encode the true contract. Note any change in the commit body.

- [ ] **Step 6: Commit**

```bash
ruff check src/skillspector/providers/host tests/unit/test_host_adapter.py
ruff format --check src/skillspector/providers/host tests/unit/test_host_adapter.py
git add src/skillspector/providers/host tests/unit/test_host_adapter.py
git commit -m "feat: add PluginLlmChatModel adapter over the Hermes host LLM"
```

---

### Task 2: `HostLLMProvider`

The `LLMProvider` that reports availability from the ContextVar and builds the Task 1 adapter.

**Files:**
- Create: `src/skillspector/providers/host/provider.py`
- Modify: `src/skillspector/providers/host/__init__.py` (export `HostLLMProvider`)
- Test: `tests/unit/test_host_provider.py`

**Interfaces:**
- Consumes: `get_host_llm` (Task 1 `_state`), `PluginLlmChatModel` (Task 1 `adapter`).
- Produces: `HostLLMProvider` with `resolve_credentials() -> None`, `is_available() -> tuple[bool, str | None]`, `get_context_length(model) -> None`, `get_max_output_tokens(model) -> None`, `resolve_model(slot="default") -> str`, `create_chat_model(model, *, max_tokens, timeout=120) -> PluginLlmChatModel | None`. Class attrs `DEFAULT_MODEL = "host"`, `SLOT_DEFAULTS: dict[str, str] = {}`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_host_provider.py` (with SPDX header):

```python
from __future__ import annotations

from skillspector.providers.host import reset_host_llm, set_host_llm
from skillspector.providers.host.adapter import PluginLlmChatModel
from skillspector.providers.host.provider import HostLLMProvider


class _FakeHostLlm:
    async def acomplete(self, **kw):  # pragma: no cover - not called here
        raise AssertionError

    async def acomplete_structured(self, **kw):  # pragma: no cover
        raise AssertionError


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_host_provider.py -q`
Expected: FAIL — `ModuleNotFoundError: skillspector.providers.host.provider`.

- [ ] **Step 3: Write minimal implementation**

`src/skillspector/providers/host/provider.py` (with SPDX header):

```python
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
        # present, else a non-empty sentinel (required by the provider protocol).
        return os.environ.get("SKILLSPECTOR_MODEL", "").strip() or self.DEFAULT_MODEL

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
        override_model = os.environ.get("SKILLSPECTOR_MODEL", "").strip() or None
        return PluginLlmChatModel(host_llm, model=override_model)
```

Update `src/skillspector/providers/host/__init__.py` to also export the provider:

```python
from ._state import get_host_llm, reset_host_llm, set_host_llm
from .provider import HostLLMProvider

__all__ = ["HostLLMProvider", "get_host_llm", "reset_host_llm", "set_host_llm"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_host_provider.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
ruff check src/skillspector/providers/host tests/unit/test_host_provider.py
ruff format --check src/skillspector/providers/host tests/unit/test_host_provider.py
git add src/skillspector/providers/host tests/unit/test_host_provider.py
git commit -m "feat: add HostLLMProvider backed by the bound host LLM"
```

---

### Task 3: Provider selection precedence + `auto`

Route to `HostLLMProvider` when the ContextVar is set (or `SKILLSPECTOR_PROVIDER=auto`), ahead of the env lookup, without changing the unset default.

**Files:**
- Modify: `src/skillspector/providers/__init__.py:76-128` (`_select_active_provider`)
- Test: `tests/unit/test_provider_selection.py`

**Interfaces:**
- Consumes: `get_host_llm`, `HostLLMProvider` (Task 2).
- Produces: `_select_active_provider()` returns `HostLLMProvider` when a host LLM is bound or `SKILLSPECTOR_PROVIDER=auto`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_provider_selection.py` (with SPDX header):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_provider_selection.py -q`
Expected: FAIL — `test_bound_host_llm_takes_precedence_over_env` and `test_auto_env_selects_host_provider` fail (currently `openai`/unknown-provider behavior).

- [ ] **Step 3: Write minimal implementation**

In `src/skillspector/providers/__init__.py`, add the import near `from .nv_build import NvBuildProvider`:

```python
from .host import HostLLMProvider, get_host_llm
```

Then insert at the very top of `_select_active_provider()` body (before `name = os.environ.get(...)`):

```python
    # A bound host LLM (Hermes plugin call) wins over any env selection.
    if get_host_llm() is not None:
        return HostLLMProvider()

    name = os.environ.get("SKILLSPECTOR_PROVIDER", "").strip().lower()

    if name == "auto":
        # Explicit opt-in to the host LLM; reports unavailable if none is bound.
        return HostLLMProvider()
```

(Keep the existing `name = ...` line only once — replace the original assignment with the block above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_provider_selection.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the provider regression tests**

Run: `python -m pytest tests/unit -k "provider" -q`
Expected: PASS — existing provider tests unaffected by the new leading branch.

- [ ] **Step 6: Commit**

```bash
ruff check src/skillspector/providers/__init__.py tests/unit/test_provider_selection.py
ruff format --check src/skillspector/providers/__init__.py tests/unit/test_provider_selection.py
git add src/skillspector/providers/__init__.py tests/unit/test_provider_selection.py
git commit -m "feat: select host provider when a host LLM is bound or provider=auto"
```

---

### Task 4: Route `get_chat_model` / `is_llm_available` to the host provider

Teach the LLM utilities to treat the host provider like the CLI providers (capability-based, before the credential path).

**Files:**
- Modify: `src/skillspector/llm_utils.py` (add `is_host_provider`; branch in `get_chat_model` and `is_llm_available`)
- Test: `tests/unit/test_llm_utils_host.py`

**Interfaces:**
- Consumes: `HostLLMProvider` (Task 2), `PluginLlmChatModel` (Task 1).
- Produces: `is_host_provider(provider: object) -> bool`; `get_chat_model()` returns a `PluginLlmChatModel` when the host provider is active; `is_llm_available()` returns `(True, None)` when a host LLM is bound.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_llm_utils_host.py` (with SPDX header):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_llm_utils_host.py -q`
Expected: FAIL — `get_chat_model` raises no-API-key (host has no credentials and isn't routed yet).

- [ ] **Step 3: Write minimal implementation**

In `src/skillspector/llm_utils.py`, add a helper after the imports:

```python
def is_host_provider(provider: object) -> bool:
    """Return ``True`` when *provider* is the Hermes host-LLM provider."""
    from skillspector.providers.host import HostLLMProvider

    return isinstance(provider, HostLLMProvider)
```

In `is_llm_available()`, add the host branch alongside the CLI branch:

```python
    provider = get_active_provider()
    if is_host_provider(provider):
        return provider.is_available()  # type: ignore[attr-defined]
    if has_cli_capability(provider):
        return provider.is_available()  # type: ignore[attr-defined]
```

In `get_chat_model()`, add the host branch before the `_resolve_default_chat_model()` line (which assumes credentials exist):

```python
    provider = get_active_provider()
    if is_host_provider(provider):
        resolved_model = model or provider.resolve_model()
        return provider.create_chat_model(
            resolved_model, max_tokens=get_max_output_tokens(resolved_model), timeout=120
        )
    if has_cli_capability(provider):
        resolved_model = model or provider.resolve_model()
        return AgentCLIChatModel(provider, resolved_model, get_max_output_tokens(resolved_model))

    model = model or _resolve_default_chat_model()
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_llm_utils_host.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
ruff check src/skillspector/llm_utils.py tests/unit/test_llm_utils_host.py
ruff format --check src/skillspector/llm_utils.py tests/unit/test_llm_utils_host.py
git add src/skillspector/llm_utils.py tests/unit/test_llm_utils_host.py
git commit -m "feat: route get_chat_model and is_llm_available to the host provider"
```

---

### Task 5: Fix the `run_scan` availability gate

Replace the credential-only gate with the capability-aware check so host (and CLI) providers are honored.

**Files:**
- Modify: `src/skillspector/mcp_server.py:36` (import) and `:77` (gate)
- Test: `tests/unit/test_mcp_server_gate.py`

**Interfaces:**
- Consumes: `is_llm_available` (Task 4).
- Produces: `run_scan(...)` verdict field `llm_available` reflects `is_llm_available()`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_mcp_server_gate.py` (with SPDX header):

```python
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
```

(If `run_scan`'s verdict assembly reads more keys off the graph result than `findings`/`filtered_findings`, extend `_FakeGraph.ainvoke`'s returned dict with those keys set to empty/neutral values so the canned state is complete.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_mcp_server_gate.py -q`
Expected: FAIL — `mcp_server` has no attribute `is_llm_available` (still importing/using `resolve_provider_credentials`).

- [ ] **Step 3: Write minimal implementation**

In `src/skillspector/mcp_server.py`, replace the import on line 36:

```python
from skillspector.llm_utils import is_llm_available
```

(Remove the now-unused `from skillspector.providers import resolve_provider_credentials` import.)

Replace the gate on line 77:

```python
    llm_available, _ = is_llm_available()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_mcp_server_gate.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Run the existing MCP server tests**

Run: `python -m pytest tests/unit/test_mcp_server.py -q`
Expected: PASS — the honest-accounting behavior is unchanged for credentialed/no-LLM cases.

- [ ] **Step 6: Commit**

```bash
ruff check src/skillspector/mcp_server.py tests/unit/test_mcp_server_gate.py
ruff format --check src/skillspector/mcp_server.py tests/unit/test_mcp_server_gate.py
git add src/skillspector/mcp_server.py tests/unit/test_mcp_server_gate.py
git commit -m "fix: gate run_scan LLM pass on capability-aware availability"
```

---

### Task 6: Rewrite the plugin handler to inject the host LLM

Drop all key/env-lock/provider/model handling; set the ContextVar around the scan; bind `ctx.llm` at call time.

**Files:**
- Modify: `extensions/hermes/skillspector_hermes/tools.py` (full rewrite of the handler body)
- Modify: `extensions/hermes/skillspector_hermes/__init__.py` (bind `ctx.llm` into the handler)
- Modify: `tests/unit/test_hermes_plugin.py` (drop provider/model cases; add host-LLM injection + reset cases)

**Interfaces:**
- Consumes: `set_host_llm` / `reset_host_llm` (`skillspector.providers.host`), `run_scan`.
- Produces: `skillspector_scan(args: object, *, host_llm: object | None = None, **_kwargs) -> str`.

- [ ] **Step 1: Write the failing test**

Edit `tests/unit/test_hermes_plugin.py`: remove `test_non_string_provider_returns_json_error` (provider is no longer a handler arg) and add:

```python
def test_host_llm_is_bound_during_scan_and_reset_after(monkeypatch) -> None:
    seen = {}

    class _FakeHostLlm:
        pass

    host = _FakeHostLlm()

    def _fake_run_scan_sync(target, *, use_llm, output_format, yara_rules_dir):
        from skillspector.providers.host import get_host_llm

        seen["bound_during"] = get_host_llm()
        return {"risk_score": 0, "safe_to_install": True}

    monkeypatch.setattr(tools, "_run_scan_sync", _fake_run_scan_sync)

    out = json.loads(tools.skillspector_scan({"target": "x", "use_llm": True}, host_llm=host))
    from skillspector.providers.host import get_host_llm

    assert seen["bound_during"] is host          # bound while the scan ran
    assert get_host_llm() is None                # reset afterwards
    assert out["safe_to_install"] is True


def test_host_llm_reset_even_when_scan_raises(monkeypatch) -> None:
    class _FakeHostLlm:
        pass

    def _boom(*a, **k):
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(tools, "_run_scan_sync", _boom)
    out = json.loads(tools.skillspector_scan({"target": "x"}, host_llm=_FakeHostLlm()))
    from skillspector.providers.host import get_host_llm

    assert get_host_llm() is None                # reset in finally
    assert "error" in out                        # never raises
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_hermes_plugin.py -q`
Expected: FAIL — `skillspector_scan` has no `host_llm` parameter yet.

- [ ] **Step 3: Write minimal implementation**

Rewrite `extensions/hermes/skillspector_hermes/tools.py` (keep the SPDX header; new module body below the header):

```python
"""Hermes Agent tool handler for SkillSpector.

The handler imports SkillSpector's framework-independent ``run_scan`` core (the
same one the MCP server wraps) so the plugin shares one scan implementation.
SkillSpector must be installed in the Hermes Python environment
(``pip install skillspector``).

The optional LLM semantic pass runs against the Hermes host model: ``register``
binds ``ctx.llm`` into the handler, which publishes it via
:func:`skillspector.providers.host.set_host_llm` for the duration of the scan.
No provider/API-key/env handling lives here — the host owns credentials.

Per the Hermes handler contract, ``skillspector_scan`` always returns a JSON
string and never raises: failures — including a missing SkillSpector install —
come back as ``{"error": ...}``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

_VALID_FORMATS = ("json", "markdown", "sarif", "terminal")


def _run_scan_sync(
    target: str, *, use_llm: bool, output_format: str, yara_rules_dir: str | None
) -> dict[str, Any]:
    """Invoke the async SkillSpector scan core from a synchronous handler."""
    from skillspector.mcp_server import run_scan

    return asyncio.run(
        run_scan(
            target,
            use_llm=use_llm,
            output_format=output_format,
            yara_rules_dir=yara_rules_dir,
        )
    )


def skillspector_scan(args: object, *, host_llm: object | None = None, **_kwargs: object) -> str:
    """Scan a target for security risks and return a JSON verdict string.

    ``host_llm`` is the Hermes ``ctx.llm``, bound by ``register`` at call time;
    when present it drives the optional LLM pass (``use_llm=true``) with no
    plugin-managed credentials. ``args`` is typed ``object`` because the
    never-raise contract requires defensively accepting any payload.
    """
    if not isinstance(args, dict):
        return json.dumps({"error": "`args` must be an object."})

    target = args.get("target")
    if not target or not isinstance(target, str):
        return json.dumps({"error": "`target` is required and must be a string."})

    output_format = args.get("output_format", "json")
    if output_format not in _VALID_FORMATS:
        return json.dumps(
            {"error": f"`output_format` must be one of {list(_VALID_FORMATS)}, got {output_format!r}."}
        )

    use_llm_arg = args.get("use_llm", False)
    if not isinstance(use_llm_arg, bool):
        return json.dumps({"error": "`use_llm` must be a boolean."})
    use_llm = use_llm_arg

    yara_rules_dir = args.get("yara_rules_dir")
    if yara_rules_dir is not None and not isinstance(yara_rules_dir, str):
        return json.dumps({"error": "`yara_rules_dir` must be a string path."})

    token = None
    try:
        if host_llm is not None:
            from skillspector.providers.host import set_host_llm

            token = set_host_llm(host_llm)

        verdict = _run_scan_sync(
            target,
            use_llm=use_llm,
            output_format=output_format,
            yara_rules_dir=yara_rules_dir,
        )
        if not isinstance(verdict, dict):
            return json.dumps(
                {
                    "error": "SkillSpector returned an unexpected (non-object) scan result.",
                    "type": type(verdict).__name__,
                }
            )
        return json.dumps(verdict, default=str)
    except Exception as exc:  # noqa: BLE001 — contract: never raise, return JSON
        if isinstance(exc, ModuleNotFoundError) and exc.name == "skillspector":
            return json.dumps(
                {
                    "error": (
                        "SkillSpector is not installed in the Hermes environment. "
                        "Install it with: pip install skillspector"
                    ),
                    "detail": str(exc),
                }
            )
        return json.dumps({"error": str(exc), "type": type(exc).__name__})
    finally:
        if token is not None:
            from skillspector.providers.host import reset_host_llm

            reset_host_llm(token)
```

Rewrite the handler wiring in `extensions/hermes/skillspector_hermes/__init__.py` `register`:

```python
def register(ctx):
    """Called once at plugin startup. Wires the scan schema to its handler.

    ``ctx.llm`` is read at call time (not registration time) so each scan uses
    the host's currently-active model, then bound into the handler.
    """

    def _handler(args, **kwargs):
        return tools.skillspector_scan(args, host_llm=ctx.llm, **kwargs)

    ctx.register_tool(
        name="skillspector_scan",
        toolset="skillspector",
        schema=schemas.SKILLSPECTOR_SCAN,
        handler=_handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_hermes_plugin.py -q`
Expected: PASS (all remaining + 2 new tests).

- [ ] **Step 5: Commit**

```bash
ruff check extensions/hermes/skillspector_hermes tests/unit/test_hermes_plugin.py
ruff format --check extensions/hermes/skillspector_hermes tests/unit/test_hermes_plugin.py
git add extensions/hermes/skillspector_hermes/tools.py extensions/hermes/skillspector_hermes/__init__.py tests/unit/test_hermes_plugin.py
git commit -m "feat: bind host ctx.llm into the scan; drop plugin key/env handling"
```

---

### Task 7: Slim the plugin schema, manifest, and docs

Remove the `provider`/`model` args and key env-vars; document host-LLM usage and the `config.yaml` override block.

**Files:**
- Modify: `extensions/hermes/skillspector_hermes/schemas.py` (drop `provider`/`model` properties)
- Modify: `extensions/hermes/skillspector_hermes/plugin.yaml` (drop key env-vars; keep only host-relevant note)
- Modify: `docs/HERMES_PLUGIN.md` (host-LLM behavior + `config.yaml` `llm:` block)
- Test: `tests/unit/test_hermes_plugin.py` (assert schema no longer advertises `provider`)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_hermes_plugin.py`:

```python
def test_schema_has_no_provider_or_model_params() -> None:
    props = schemas.SKILLSPECTOR_SCAN["parameters"]["properties"]
    assert "provider" not in props
    assert "model" not in props
    assert set(props) == {"target", "use_llm", "output_format", "yara_rules_dir"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_hermes_plugin.py::test_schema_has_no_provider_or_model_params -q`
Expected: FAIL — `provider`/`model` still present.

- [ ] **Step 3: Write minimal implementation**

In `schemas.py`, delete the `"provider"` and `"model"` entries from `properties` (keep `target`, `use_llm`, `output_format`, `yara_rules_dir`; `required` stays `["target"]`). Update the `use_llm` description to: `"Run the optional LLM semantic pass (uses the Hermes host model). Defaults to false (fast static-only scan)."`

Replace `plugin.yaml`'s `optional_env:` block with just the host-relevant note (drop `SKILLSPECTOR_PROVIDER` key-list wording and all `OPENAI_*`/`ANTHROPIC_*`/`NVIDIA_INFERENCE_KEY` entries):

```yaml
optional_env:
  - name: SKILLSPECTOR_MODEL
    description: >-
      Optional model override for the host LLM pass. Leave unset to use whatever
      model the Hermes agent is currently configured with.
```

Add an operator-configuration section to `docs/HERMES_PLUGIN.md` documenting that `use_llm=true` runs against the host model with no plugin credentials, and the gated override block:

````markdown
## LLM configuration

The optional semantic pass (`use_llm: true`) runs against the **host's**
configured model via `ctx.llm` — the plugin manages no API keys. By default it
"uses what the agent is using." Operators may gate provider/model overrides per
plugin in `config.yaml`:

```yaml
plugins:
  entries:
    skillspector:
      llm:
        allow_provider_override: true
        allowed_providers: [anthropic, openrouter]
        allow_model_override: true
        allowed_models: [anthropic/claude-3-5-haiku]
```
````

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_hermes_plugin.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + lint gate**

Run: `python -m pytest -m "not integration" -q && ruff check src tests extensions && ruff format --check src tests extensions`
Expected: PASS / no findings.

- [ ] **Step 6: Commit**

```bash
git add extensions/hermes/skillspector_hermes/schemas.py extensions/hermes/skillspector_hermes/plugin.yaml docs/HERMES_PLUGIN.md tests/unit/test_hermes_plugin.py
git commit -m "feat: host-LLM plugin surface — drop provider/model args and key env"
```

---

## Self-Review

**Spec coverage:**
- `HostLLMProvider` subpackage → Task 2. Adapter with native `acomplete_structured` → Task 1. ✓
- ContextVar injection → Task 1 (`_state.py`), consumed Task 3/6. ✓
- `auto`/precedence selection; unset unchanged → Task 3. ✓
- Gate fix to `is_llm_available()` → Task 5. ✓
- `get_chat_model`/`is_llm_available` host routing → Task 4. ✓
- Plugin shrink (tools/init/schema/yaml/docs) → Task 6 + 7. ✓
- Metadata returns `None`; `resolve_model` non-empty sentinel → Task 2. ✓
- Standalone unchanged → Task 3 Step 5 + Task 5 Step 5 regression runs. ✓
- No `agent`-package runtime import; duck-typed host → Global Constraints + Task 1 `_state.py`. ✓
- Testing per spec → each task's tests + Task 7 Step 5 full suite. ✓

**Open items carried from the spec (not blockers):**
- Exact `PluginLlm` signatures/return shape — Task 1 Step 5 validates against the real class; the fake encodes the assumed contract.
- Operator override *forwarding* is plumbed (`PluginLlmChatModel(provider=, model=)`) but the handler doesn't source values from plugin config yet (how a plugin reads its own `config.yaml` block was undocumented / 404). Wiring that source is deferred, consistent with the spec's YAGNI scope.

**Placeholder scan:** none — every code step contains full content.

**Type consistency:** `set_host_llm`/`get_host_llm`/`reset_host_llm`, `HostLLMProvider`, `PluginLlmChatModel(host_llm, *, provider, model)`, `is_host_provider`, and `is_llm_available() -> (bool, str|None)` are used consistently across Tasks 1–6.
