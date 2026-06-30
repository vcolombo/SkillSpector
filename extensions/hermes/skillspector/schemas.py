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

"""Hermes Agent tool schema for SkillSpector scanning."""

SKILLSPECTOR_SCAN = {
    "name": "skillspector_scan",
    "description": (
        "Scan an AI agent skill, MCP server, or repository for security risks "
        "BEFORE installing or trusting it. Accepts a Git URL, file URL, .zip, "
        ".md file (e.g. SKILL.md), or local directory as `target`. Returns a "
        "verdict with risk_score (0-100), severity, recommendation, "
        "safe_to_install, and findings. The llm_used / scan_mode fields report "
        "whether the optional semantic LLM pass actually ran, so a low score "
        "from a static-only scan is not mistaken for a clean full scan."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Path, URL, zip, Git repo, or SKILL.md file to scan.",
            },
            "use_llm": {
                "type": "boolean",
                "description": (
                    "Run the optional LLM semantic pass on top of static "
                    "analysis. Honoured only when provider credentials resolve. "
                    "Defaults to false (fast static-only scan)."
                ),
            },
            "output_format": {
                "type": "string",
                "enum": ["json", "markdown", "sarif", "terminal"],
                "description": "Format of the embedded `report` string. Defaults to json.",
            },
            "yara_rules_dir": {
                "type": "string",
                "description": "Optional directory of additional YARA rules.",
            },
            "provider": {
                "type": "string",
                "enum": ["openai", "anthropic", "anthropic_proxy", "nv_build", "nv_inference"],
                "description": (
                    "Optional LLM provider, used only when use_llm is true. "
                    "Overrides SKILLSPECTOR_PROVIDER for this scan."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model override, used only when use_llm is true.",
            },
        },
        "required": ["target"],
    },
}
