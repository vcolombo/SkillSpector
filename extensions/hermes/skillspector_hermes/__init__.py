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

"""SkillSpector Hermes Agent plugin registration entry point.

Registers the ``skillspector_scan`` tool under the ``skillspector`` toolset so
a Hermes agent can vet a skill, MCP server, or repository and gate installs on
the returned risk verdict.

The package is named ``skillspector_hermes`` rather than ``skillspector`` on
purpose: Hermes imports a plugin by its directory name, and a package named
``skillspector`` would shadow the installed ``skillspector`` distribution that
:mod:`tools` imports at scan time (``from skillspector.mcp_server import
run_scan``), breaking every scan. The ``skillspector`` *toolset* below is just a
display namespace and does not collide.
"""

from . import schemas, tools


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
