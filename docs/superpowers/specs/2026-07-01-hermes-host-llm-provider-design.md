# Design: `auto` host-LLM provider + Hermes plugin rewrite

- **Date:** 2026-07-01
- **Branch:** `feat/hermes-agent-plugin`
- **Status:** Approved (brainstorm) → pending spec review

## Problem

SkillSpector ships as a Hermes Agent plugin (`extensions/hermes/skillspector_hermes/`),
currently implemented as a **Tool** that shells/imports `run_scan` and manages its own
LLM credentials (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) via a process-`env`
save/restore dance guarded by `_ENV_LOCK`.

That design duplicates the CLI's own env-var provider resolution and earned nothing for
it — seven rounds of review churn came almost entirely from the key/env-lock surface.
It also forfeits the single most valuable Hermes integration: the host already has a
configured LLM (`ctx.llm`), and Hermes's posture is *"use what the user is using."*

Hermes's own guidance: a capability that needs **API-key management or custom Python
integration** is a **Tool** (not a declarative Skill), and Tools reach the host model
through `ctx.llm` — which only Python `register(ctx)` code can access. So the LLM-reuse
goal *requires* a Tool; a `SKILL.md` cannot touch `ctx.llm`.

## Goal

Let SkillSpector's optional LLM semantic pass run against **the host's already-configured
model** when invoked as a Hermes plugin, with **no credentials managed by the plugin**,
while leaving standalone CLI / non-Hermes MCP behavior byte-for-byte unchanged. Target:
an upstream-quality feature mergeable into NVIDIA's SkillSpector.

## Key facts grounding the design

- The LLM analyzers (`llm_analyzer_base.py`) only ever use a chat model via
  `get_chat_model()` → `.with_structured_output(schema)` → `.invoke()/.ainvoke()`, plus a
  plain `.invoke()` text path. That is the *entire* surface an adapter must implement.
- `AgentCLIChatModel` (`llm_utils.py`) already implements exactly that surface for a
  non-HTTP transport (the local `claude`/`codex`/`gemini` CLIs). The host adapter is its
  sibling.
- `ctx.llm` is an `agent.plugin_llm.PluginLlm` with async-native methods
  `acomplete(messages=…, purpose=…)` and
  `acomplete_structured(instructions=…, input=…, json_schema=…, purpose=…)`. The host
  owns keys, provider resolution, timeouts, fallback, and audit attribution
  (`result.audit`). *"No keys, no provider config, no SDK initialisation."*
- Operator provider/model overrides are gated **host-side** in `config.yaml` under
  `plugins.entries.<plugin>.llm` (`allow_provider_override`, `allowed_providers`,
  `allow_model_override`, `allowed_models`). The plugin forwards `provider=`/`model=` into
  `ctx.llm` only when configured; passing them without opt-in raises `PluginLlmTrustError`.
- **Existing gate bug:** `run_scan` computes
  `llm_available = resolve_provider_credentials() is not None`. Credential-less providers
  (the CLIs, and now the host provider) are wrongly judged "unavailable" under MCP.

## Architecture

### 1. `HostLLMProvider` — new provider subpackage

`src/skillspector/providers/host/` (`__init__.py`, `provider.py`, adapter module):

- Conforms to `LLMProvider` (`ModelMetadataProvider` + `CredentialsProvider` +
  `ChatModelProvider`).
- `resolve_credentials() -> None` — the host owns keys; availability is *not* credential-based.
- `is_available() -> tuple[bool, str | None]` — `(True, None)` iff the host-LLM
  `ContextVar` is set; otherwise `(False, "<reason>")`. This marks it as a
  capability-based provider, analogous to `AgentCLICapable`, but **distinct** so it does
  not get routed to the CLI prompt-and-parse adapter.
- Metadata: `get_context_length()/get_max_output_tokens() -> None` ("I don't know" →
  callers fall back to defaults). `resolve_model()` returns a non-empty `"host"` sentinel
  (the real model is chosen host-side); `DEFAULT_MODEL = "host"`, `SLOT_DEFAULTS = {}`.
- `create_chat_model(model, *, max_tokens, timeout)` returns the `PluginLlmChatModel`
  adapter bound to the host LLM from the ContextVar.

### 2. `PluginLlmChatModel` — LangChain-compatible adapter

Mirrors `AgentCLIChatModel`; implements only `invoke`/`ainvoke`/`with_structured_output`
and stubs `batch`/`stream` to raise `NotImplementedError` (explicit boundary).

- **Text path:** `await ctx.llm.acomplete(messages=[…], purpose="skillspector-scan")`;
  sync `invoke` bridges via `asyncio.run`/thread as needed (analyzers use `ainvoke` on the
  async scan path).
- **Structured path:** `with_structured_output(schema)` returns an object whose
  `ainvoke(prompt)` calls
  `ctx.llm.acomplete_structured(instructions=…, input=…, json_schema=schema.model_json_schema(), purpose="skillspector-scan")`
  and validates the result into `schema` (native host structured output — more robust than
  emit-raw-JSON-and-parse).
- `max_tokens=None` (host default). Optional `provider=`/`model=` are forwarded **only**
  when an operator override is configured; `PluginLlmTrustError` is caught and the call
  retried against the host default.

### 3. Injection via `ContextVar`

A module-level `ContextVar[PluginLlm | None]` in `providers/host/`. The plugin handler
`set()`s it before `run_scan` and `reset()`s it in `finally`. Async-safe; no function
signatures threaded through the LangGraph (`graph.ainvoke`) that sits between `run_scan`
and the analyzers, and **no process-env mutation** (so `_ENV_LOCK` and env save/restore
are removed).

### 4. Provider selection precedence

`_select_active_provider()` gains a leading rule:

```
if get_host_llm() is not None:          # ContextVar set → running under Hermes
    return HostLLMProvider()
name = os.environ.get("SKILLSPECTOR_PROVIDER", "").strip().lower()
if name == "auto":                       # optional standalone alias
    return HostLLMProvider()             # reports unavailable if no host LLM set
... existing branches unchanged ...
```

Env **unset** still resolves to `nv_inference`/`nv_build` exactly as today.

### 5. Gate fix

`run_scan` replaces `resolve_provider_credentials() is not None` with the capability-aware
availability check (`is_llm_available()`), so host and CLI providers are honored. The
honest `llm_requested/llm_available/llm_used/scan_mode` accounting in the returned verdict
is retained unchanged.

`get_chat_model()` gains a host branch (before `_resolve_default_chat_model()`, which
assumes credentials) that returns the host provider's `create_chat_model()` output —
parallel to the existing `has_cli_capability` branch.

### 6. Plugin surface

- `__init__.py`: `register(ctx)` binds `ctx` into the handler by closure.
- `tools.py`: drops all key/env-lock logic; handler sets/resets the ContextVar around
  `run_scan`; the never-raise contract (always return a JSON string) is kept, including a
  `finally` that resets the ContextVar even on error.
- `schemas.py`: provider enum removed (host owns provider choice).
- `plugin.yaml`: `optional_env` loses `OPENAI_API_KEY`/`ANTHROPIC_*`/`NVIDIA_INFERENCE_KEY`
  etc. Operator provider/model overrides are documented as the `config.yaml`
  `plugins.entries.skillspector.llm:` block.

## Error handling

- Handler: never raises; returns a JSON error verdict; ContextVar reset in `finally`.
- Adapter: fail-closed — if `ctx.llm` errors/times out, the analyzer's existing handling
  applies and the scan reports `llm_used` honestly (no silent success).
- `PluginLlmTrustError` is only reachable when forwarding a gated override; caught and
  downgraded to host-default.

## Standalone behavior (unchanged)

CLI and non-Hermes MCP with `SKILLSPECTOR_PROVIDER` unset resolve to `nv_build`/
`nv_inference` and use env-var credentials, identical to today. The ContextVar is never set
outside the plugin, so the host branch is inert.

## Testing

Unit (`pytest -m "not integration"`, ruff + ruff format, line-length 100):

- ContextVar drives selection and takes precedence over `SKILLSPECTOR_PROVIDER`.
- `HostLLMProvider.is_available()` reflects ContextVar presence; `resolve_credentials()` is
  `None`; metadata methods return `None`; `resolve_model()` non-empty.
- Adapter structured + text paths against a fake `PluginLlm` double; schema validation on
  the structured path; `PluginLlmTrustError` fallback.
- `run_scan` reports `llm_available/llm_used` True with host present, static-only when
  absent (gate fix).
- Plugin handler: binds `ctx`, sets/resets ContextVar with no leak on error, returns JSON,
  never raises — reusing the existing importlib-load harness in
  `tests/unit/test_hermes_plugin.py`.
- Regression: unset provider still `nv_build`; existing provider/CLI tests unaffected.

## Out of scope (YAGNI)

Streaming, vision-model routing, and batch — the adapter stubs them to fail loudly, as
`AgentCLIChatModel` already does. Operator override **forwarding** is plumbed but defaults
to off (pure host-default), pending confirmation of how a plugin reads its own config block.

## Open question

The Hermes `creating-plugins` doc page returned 404, so the exact mechanism by which a
plugin reads its own `config.yaml` block (to source an override value to forward) is
unverified. Override-forwarding is therefore scoped as optional; confirm against a real
Hermes plugin example during implementation planning.
