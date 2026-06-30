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
``extensions/hermes/skillspector/``) and is named ``skillspector`` itself, so it
is loaded here by file path under a unique module name to avoid shadowing the
real package. These tests cover the handler layer's own logic — argument
validation, env-override save/restore, and the "always return JSON, never raise"
contract — not the scan core (covered by ``test_mcp_server.py``).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

_PLUGIN_TOOLS = (
    Path(__file__).resolve().parents[2] / "extensions" / "hermes" / "skillspector" / "tools.py"
)


def _load_tools() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hermes_skillspector_tools", _PLUGIN_TOOLS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tools = _load_tools()


def test_missing_target_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({}))
    assert "error" in out and "target" in out["error"]


def test_non_string_target_returns_json_error() -> None:
    out = json.loads(tools.skillspector_scan({"target": 123}))
    assert "error" in out


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
