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

"""Hermes Agent tool handlers for SkillSpector.

The handler imports SkillSpector's framework-independent ``run_scan`` core
(the same one the MCP server wraps) so the Hermes plugin shares one scan
implementation rather than reimplementing it. SkillSpector must be installed
in the Hermes Python environment (``pip install skillspector``).

Per the Hermes handler contract, ``skillspector_scan`` always returns a JSON
string and never raises: failures — including a missing SkillSpector install —
come back as ``{"error": ...}``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

_VALID_FORMATS = ("json", "markdown", "sarif", "terminal")
_VALID_PROVIDERS = ("openai", "anthropic", "anthropic_proxy", "nv_build", "nv_inference")


def _run_scan_sync(target: str, *, use_llm: bool, output_format: str, yara_rules_dir: str | None) -> dict[str, Any]:
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


def skillspector_scan(args: dict, **kwargs) -> str:
    """Scan a target for security risks and return a JSON verdict string."""
    target = args.get("target")
    if not target or not isinstance(target, str):
        return json.dumps({"error": "`target` is required and must be a string."})

    output_format = args.get("output_format", "json")
    if output_format not in _VALID_FORMATS:
        return json.dumps(
            {"error": f"`output_format` must be one of {list(_VALID_FORMATS)}, got {output_format!r}."}
        )

    use_llm = bool(args.get("use_llm", False))

    provider = args.get("provider")
    if provider is not None and provider not in _VALID_PROVIDERS:
        return json.dumps(
            {"error": f"`provider` must be one of {list(_VALID_PROVIDERS)}, got {provider!r}."}
        )

    yara_rules_dir = args.get("yara_rules_dir")
    if yara_rules_dir is not None and not isinstance(yara_rules_dir, str):
        return json.dumps({"error": "`yara_rules_dir` must be a string path."})

    # Provider/model selection is resolved by SkillSpector from the environment.
    # Apply per-call overrides only when an LLM pass is actually requested.
    saved_env: dict[str, str | None] = {}
    if use_llm:
        if provider:
            saved_env["SKILLSPECTOR_PROVIDER"] = os.environ.get("SKILLSPECTOR_PROVIDER")
            os.environ["SKILLSPECTOR_PROVIDER"] = provider
        model = args.get("model")
        if model:
            saved_env["SKILLSPECTOR_MODEL"] = os.environ.get("SKILLSPECTOR_MODEL")
            os.environ["SKILLSPECTOR_MODEL"] = str(model)

    try:
        verdict = _run_scan_sync(
            target,
            use_llm=use_llm,
            output_format=output_format,
            yara_rules_dir=yara_rules_dir,
        )
        return json.dumps(verdict, default=str)
    except ModuleNotFoundError as exc:
        return json.dumps(
            {
                "error": (
                    "SkillSpector is not installed in the Hermes environment. "
                    "Install it with: pip install skillspector"
                ),
                "detail": str(exc),
            }
        )
    except Exception as exc:  # noqa: BLE001 — contract: never raise, return JSON
        return json.dumps({"error": str(exc), "type": type(exc).__name__})
    finally:
        for key, prior in saved_env.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
