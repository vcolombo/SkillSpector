# SkillSpector Hermes Agent Plugin

SkillSpector ships as a [Hermes Agent](https://hermes-agent.nousresearch.com/docs/guides/build-a-hermes-plugin)
plugin. It registers a `skillspector_scan` tool (under the `skillspector`
toolset) so a Hermes agent can vet a skill, MCP server, or repository **before**
installing it and gate the install on the returned risk verdict.

The plugin reuses SkillSpector's framework-independent scan core (`run_scan`,
the same one the [MCP server](../src/skillspector/mcp_server.py) wraps) — it does
not reimplement scanning.

## Requirements

- Hermes Agent installed.
- Python `>=3.12,<3.15`.
- This repository checked out locally — the plugin directory
  (`extensions/hermes/skillspector_hermes/`) ships here and is copied into your
  Hermes plugins directory below.
- SkillSpector installed in the **same** Python environment Hermes runs in:

  ```bash
  pip install skillspector
  # or: uv pip install skillspector
  ```

## Install

Copy the bundled plugin directory into your Hermes plugins directory:

```bash
mkdir -p ~/.hermes/plugins
cp -r extensions/hermes/skillspector_hermes ~/.hermes/plugins/skillspector_hermes
hermes plugins enable skillspector_hermes
```

> The plugin package is named `skillspector_hermes` (not `skillspector`) so it
> cannot shadow the installed `skillspector` distribution it imports at scan
> time. The tool it exposes is still `skillspector_scan`.

Verify discovery (set `HERMES_PLUGINS_DEBUG=1` for verbose logging):

```bash
hermes plugins list
```

## Basic scan

Ask Hermes:

```text
Use skillspector_scan on ./my-skill before I install it.
```

The tool returns a JSON verdict:

```json
{
  "target": "./my-skill",
  "risk_score": 93,
  "severity": "CRITICAL",
  "recommendation": "...",
  "safe_to_install": false,
  "findings": [ ... ],
  "report": "...",
  "llm_requested": false,
  "llm_available": false,
  "llm_used": false,
  "scan_mode": "static-only",
  "version": "2.3.7"
}
```

`safe_to_install` is `false` when `risk_score > 50` (mirroring the CLI's exit
code). The `llm_used` / `scan_mode` fields report whether the optional semantic
pass actually ran, so a low score from a static-only scan is never mistaken for
a clean full scan.

## Tool parameters

- `target` (required): path, URL, zip, Git repo, or `SKILL.md` file to scan.
- `use_llm`: run the optional semantic LLM pass (uses the Hermes host model).
  Default `false` (fast, static-only).
- `output_format`: `json`, `markdown`, `sarif`, or `terminal`. Default `json`.
- `yara_rules_dir`: optional directory of additional YARA rules.

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

`SKILLSPECTOR_MODEL` (declared as `optional_env` in `plugin.yaml`) is an
optional model override for the host LLM pass; leave it unset to use whatever
model the Hermes agent is currently configured with. Static analysis is the
default and needs no LLM configuration at all — omit `use_llm` (or pass
`use_llm=false`) to run static-only.

## Handler contract

`skillspector_scan` always returns a JSON string and never raises. Failures —
including SkillSpector not being installed in the Hermes environment — come back
as `{"error": ...}`, per the Hermes handler contract.

## Remove

```bash
hermes plugins disable skillspector_hermes
rm -rf ~/.hermes/plugins/skillspector_hermes
```
