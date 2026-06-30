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

"""Tests for the Hermes Agent plugin handler.

The plugin lives outside the importable ``skillspector`` package (under
``extensions/hermes/skillspector_hermes/``). Its ``tools.py`` is loaded here by
file path under a unique module name so these tests don't depend on Hermes's
loader. They cover the handler layer's own logic — argument validation,
env-override save/restore, and the "always return JSON, never raise" contract —
not the scan core (covered by ``test_mcp_server.py``).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "extensions" / "hermes"
_PKG_DIR = _PLUGIN_DIR / "skillspector_hermes"


def _load_module(filename: str, modname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(modname, _PKG_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tools = _load_module("tools.py", "hermes_skillspector_tools")
schemas = _load_module("schemas.py", "hermes_skillspector_schemas")


def test_missing_target_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({}))
    assert "error" in out and "target" in out["error"]


def test_non_string_target_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({"target": 123}))
    assert "error" in out


@pytest.mark.parametrize("bad_args", [None, [], "target", 42])
def test_non_dict_args_returns_json_error_never_raises(bad_args: object) -> None:
    # The handler must honour the never-raise contract for any payload shape.
    out = json.loads(tools.skillspector_scan(bad_args))
    assert out == {"error": "`args` must be an object."}


def test_invalid_output_format_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({"target": "x", "output_format": "xml"}))
    assert "error" in out and "output_format" in out["error"]


def test_non_bool_use_llm_is_rejected_not_coerced() -> None:
    # bool("false") is True; the handler must reject the string instead.
    out = json.loads(tools.skillspector_scan({"target": "x", "use_llm": "false"}))
    assert out == {"error": "`use_llm` must be a boolean."}


def test_non_string_provider_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({"target": "x", "use_llm": True, "provider": 5}))
    assert "error" in out and "provider" in out["error"]


def test_non_string_yara_rules_dir_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({"target": "x", "yara_rules_dir": 5}))
    assert "error" in out and "yara_rules_dir" in out["error"]


def test_env_overrides_are_restored_after_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider/model env vars are set during the scan and restored afterwards."""
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "preexisting")
    monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)

    seen: dict[str, str | None] = {}

    def fake_run(target, *, use_llm, output_format, yara_rules_dir):  # noqa: ANN001, ANN202
        seen["provider"] = os.environ.get("SKILLSPECTOR_PROVIDER")
        seen["model"] = os.environ.get("SKILLSPECTOR_MODEL")
        return {"target": target, "risk_score": 0}

    monkeypatch.setattr(tools, "_run_scan_sync", fake_run)

    out = json.loads(
        tools.skillspector_scan(
            {"target": "x", "use_llm": True, "provider": "anthropic", "model": "claude-x"}
        )
    )
    assert out["risk_score"] == 0
    # Overrides were visible to the scan...
    assert seen == {"provider": "anthropic", "model": "claude-x"}
    # ...and the prior environment is restored afterwards.
    assert os.environ.get("SKILLSPECTOR_PROVIDER") == "preexisting"
    assert "SKILLSPECTOR_MODEL" not in os.environ


def test_env_is_not_touched_when_llm_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider/model are ignored (and env untouched) when use_llm is false."""
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "preexisting")

    def fake_run(target, *, use_llm, output_format, yara_rules_dir):  # noqa: ANN001, ANN202
        assert os.environ.get("SKILLSPECTOR_PROVIDER") == "preexisting"
        return {"target": target, "risk_score": 0}

    monkeypatch.setattr(tools, "_run_scan_sync", fake_run)

    out = json.loads(
        tools.skillspector_scan({"target": "x", "use_llm": False, "provider": "anthropic"})
    )
    assert out["risk_score"] == 0
    assert os.environ.get("SKILLSPECTOR_PROVIDER") == "preexisting"


def test_missing_skillspector_is_reported_as_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise ModuleNotFoundError("No module named 'skillspector'", name="skillspector")

    monkeypatch.setattr(tools, "_run_scan_sync", boom)
    out = json.loads(tools.skillspector_scan({"target": "x"}))
    assert "not installed" in out["error"].lower()


def test_unrelated_missing_module_is_not_misreported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different missing dependency must not be reported as SkillSpector missing."""

    def boom(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise ModuleNotFoundError("No module named 'totally_unrelated'", name="totally_unrelated")

    monkeypatch.setattr(tools, "_run_scan_sync", boom)
    out = json.loads(tools.skillspector_scan({"target": "x"}))
    assert "not installed" not in out["error"].lower()
    assert out["type"] == "ModuleNotFoundError"


def test_missing_submodule_is_not_reported_as_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial/broken install (missing submodule) must not read as 'not installed'."""

    def boom(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise ModuleNotFoundError(
            "No module named 'skillspector.mcp_server'", name="skillspector.mcp_server"
        )

    monkeypatch.setattr(tools, "_run_scan_sync", boom)
    out = json.loads(tools.skillspector_scan({"target": "x"}))
    assert "not installed" not in out["error"].lower()
    assert out["type"] == "ModuleNotFoundError"


def test_arbitrary_scan_failure_returns_json_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(tools, "_run_scan_sync", boom)
    out = json.loads(tools.skillspector_scan({"target": "x"}))
    assert out == {"error": "scan exploded", "type": "RuntimeError"}


def test_static_scan_against_safe_fixture_is_clean() -> None:
    """End-to-end static scan of the bundled safe fixture returns a clean verdict."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "safe_skill"
    out = json.loads(tools.skillspector_scan({"target": str(fixture), "output_format": "json"}))
    assert out["risk_score"] == 0
    assert out["safe_to_install"] is True
    assert out["scan_mode"] == "static-only"


def test_provider_enum_is_limited_to_credential_backed_providers() -> None:
    """The schema must only advertise providers whose creds enable run_scan's LLM pass.

    run_scan gates ``llm_used`` on ``resolve_provider_credentials() is not None``,
    which is None for bedrock (SigV4) and the CLI providers — selecting those
    could never turn the semantic pass on, so they must not be offered here.
    """
    enum = schemas.SKILLSPECTOR_SCAN["parameters"]["properties"]["provider"]["enum"]
    assert set(enum) == {"openai", "anthropic", "anthropic_proxy", "nv_build", "nv_inference"}
    for excluded in ("bedrock", "claude_cli", "codex_cli", "gemini_cli", "antigravity_cli"):
        assert excluded not in enum


def test_llm_scan_holds_lock_even_without_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM-requested scan must serialize on _ENV_LOCK even with no override.

    Otherwise a concurrent call reading the ambient SKILLSPECTOR_PROVIDER/MODEL
    could observe another call's temporary override.
    """
    held: dict[str, bool] = {}

    def fake_run(target, *, use_llm, output_format, yara_rules_dir):  # noqa: ANN001, ANN202
        # If the lock is held by this call, a non-blocking acquire fails.
        got = tools._ENV_LOCK.acquire(blocking=False)
        held["during_scan"] = not got
        if got:
            tools._ENV_LOCK.release()
        return {"target": target, "risk_score": 0}

    monkeypatch.setattr(tools, "_run_scan_sync", fake_run)
    # use_llm=True, no provider/model override.
    json.loads(tools.skillspector_scan({"target": "x", "use_llm": True}))
    assert held["during_scan"] is True
    # And the lock is free again afterwards.
    assert tools._ENV_LOCK.acquire(blocking=False) is True
    tools._ENV_LOCK.release()


def test_static_scan_does_not_hold_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A static scan (use_llm=false) must not serialize on _ENV_LOCK."""
    held: dict[str, bool] = {}

    def fake_run(target, *, use_llm, output_format, yara_rules_dir):  # noqa: ANN001, ANN202
        got = tools._ENV_LOCK.acquire(blocking=False)
        held["during_scan"] = not got
        if got:
            tools._ENV_LOCK.release()
        return {"target": target, "risk_score": 0}

    monkeypatch.setattr(tools, "_run_scan_sync", fake_run)
    json.loads(tools.skillspector_scan({"target": "x", "use_llm": False}))
    assert held["during_scan"] is False


def test_override_lock_is_released_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sequential override scans must both complete (lock released, no deadlock)."""

    def fake_run(target, *, use_llm, output_format, yara_rules_dir):  # noqa: ANN001, ANN202
        return {"target": target, "risk_score": 0}

    monkeypatch.setattr(tools, "_run_scan_sync", fake_run)
    args = {"target": "x", "use_llm": True, "provider": "anthropic", "model": "m"}
    first = json.loads(tools.skillspector_scan(dict(args)))
    second = json.loads(tools.skillspector_scan(dict(args)))
    assert first["risk_score"] == 0 and second["risk_score"] == 0
    # The lock is free again after both calls returned.
    assert tools._ENV_LOCK.acquire(blocking=False) is True
    tools._ENV_LOCK.release()


def test_plugin_import_by_name_does_not_shadow_skillspector() -> None:
    """Importing the plugin the way Hermes does must not shadow the real package.

    The plugin package is deliberately named ``skillspector_hermes`` (not
    ``skillspector``). This reproduces Hermes's discovery — prepend the plugins
    directory to ``sys.path`` and import the package by name — and asserts that
    the installed ``skillspector`` distribution (and its ``mcp_server``) is still
    importable, so the lazy ``from skillspector.mcp_server import run_scan`` the
    handler relies on keeps working.
    """
    for name in ("skillspector_hermes", "skillspector_hermes.tools", "skillspector_hermes.schemas"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(_PLUGIN_DIR))
    try:
        plugin = importlib.import_module("skillspector_hermes")

        # The real distribution is untouched: skillspector resolves to src, not
        # the plugin directory, and its mcp_server (which run_scan lives in) loads.
        import skillspector
        from skillspector import mcp_server

        assert "extensions/hermes" not in str(Path(skillspector.__file__).resolve())
        assert hasattr(mcp_server, "run_scan")

        # register() wires skillspector_scan onto a Hermes-style context.
        registered: dict[str, object] = {}

        class _Ctx:
            def register_tool(self, **kwargs: object) -> None:
                registered.update(kwargs)

        plugin.register(_Ctx())
        assert registered["name"] == "skillspector_scan"
        assert registered["toolset"] == "skillspector"
        assert callable(registered["handler"])
    finally:
        sys.path.remove(str(_PLUGIN_DIR))
        for name in (
            "skillspector_hermes",
            "skillspector_hermes.tools",
            "skillspector_hermes.schemas",
        ):
            sys.modules.pop(name, None)
