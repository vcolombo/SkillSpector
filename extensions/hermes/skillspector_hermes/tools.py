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
import threading
from typing import Any

_VALID_FORMATS = ("json", "markdown", "sarif", "terminal")

# Provider/model selection is plumbed through process-global env vars that
# SkillSpector's core reads at scan time. Any LLM-requested scan therefore reads
# shared state, so concurrent invocations could leak one call's provider/model
# into another's scan. Serialize every LLM-requested scan (not just ones that
# apply an override) so a reader can never observe another call's mutation.
_ENV_LOCK = threading.Lock()


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


def skillspector_scan(args: object, **_kwargs: object) -> str:
    """Scan a target for security risks and return a JSON verdict string.

    ``args`` is typed ``object`` rather than ``dict`` because the never-raise
    contract means it must defensively accept any payload; it is narrowed to a
    mapping right below. ``_kwargs`` is required by the Hermes handler contract
    for forward compatibility but is currently unused.
    """
    # Honour the never-raise contract even if a caller passes a non-object
    # payload (None, list, ...) — args.get below would otherwise AttributeError.
    if not isinstance(args, dict):
        return json.dumps({"error": "`args` must be an object."})

    target = args.get("target")
    if not target or not isinstance(target, str):
        return json.dumps({"error": "`target` is required and must be a string."})

    output_format = args.get("output_format", "json")
    if output_format not in _VALID_FORMATS:
        return json.dumps(
            {
                "error": f"`output_format` must be one of {list(_VALID_FORMATS)}, got {output_format!r}."
            }
        )

    # Validate the actual type rather than coercing: bool("false") is True, so
    # a string slipping through here could silently enable the LLM pass.
    use_llm_arg = args.get("use_llm", False)
    if not isinstance(use_llm_arg, bool):
        return json.dumps({"error": "`use_llm` must be a boolean."})
    use_llm = use_llm_arg

    provider = args.get("provider")
    if provider is not None and not isinstance(provider, str):
        return json.dumps({"error": "`provider` must be a string."})
    # The set of valid providers is owned by SkillSpector's core, which raises a
    # clear error for unknown values (surfaced below as a JSON error). We do not
    # duplicate that whitelist here to avoid drifting out of sync with it.

    model = args.get("model")
    if model is not None and not isinstance(model, str):
        return json.dumps({"error": "`model` must be a string."})

    yara_rules_dir = args.get("yara_rules_dir")
    if yara_rules_dir is not None and not isinstance(yara_rules_dir, str):
        return json.dumps({"error": "`yara_rules_dir` must be a string path."})

    # Provider/model selection is resolved by SkillSpector from the environment.
    # Hold the env lock for the whole scan whenever the LLM pass is requested:
    # the scan reads SKILLSPECTOR_PROVIDER/MODEL even when this call passes no
    # override, so a concurrent override elsewhere would otherwise leak in.
    # Static scans (use_llm=false) don't depend on these and stay concurrent.
    saved_env: dict[str, str | None] = {}
    lock_held = False
    if use_llm:
        _ENV_LOCK.acquire()
        lock_held = True
        if provider:
            saved_env["SKILLSPECTOR_PROVIDER"] = os.environ.get("SKILLSPECTOR_PROVIDER")
            os.environ["SKILLSPECTOR_PROVIDER"] = provider
        if model:
            saved_env["SKILLSPECTOR_MODEL"] = os.environ.get("SKILLSPECTOR_MODEL")
            os.environ["SKILLSPECTOR_MODEL"] = model

    try:
        verdict = _run_scan_sync(
            target,
            use_llm=use_llm,
            output_format=output_format,
            yara_rules_dir=yara_rules_dir,
        )
        return json.dumps(verdict, default=str)
    except Exception as exc:  # noqa: BLE001 — contract: never raise, return JSON
        # Special-case only the top-level `skillspector` package being absent as
        # "not installed". A missing *submodule* (exc.name like
        # "skillspector.mcp_server") signals a partial/broken or version-mismatched
        # install, and an unrelated missing dependency is different again — both
        # fall through to the generic error below with their real detail.
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
        try:
            for key, prior in saved_env.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior
        finally:
            # Always release after restoring the environment, never before, so a
            # concurrent waiter resumes against the restored env (not a deadlock).
            if lock_held:
                _ENV_LOCK.release()
