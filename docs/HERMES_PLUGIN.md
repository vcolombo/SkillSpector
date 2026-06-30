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
- `use_llm`: run the optional semantic LLM pass. Default `false` (fast,
  static-only). Honoured only when provider credentials resolve.
- `output_format`: `json`, `markdown`, `sarif`, or `terminal`. Default `json`.
- `yara_rules_dir`: optional directory of additional YARA rules.
- `provider`: optional LLM provider (`openai`, `anthropic`, `anthropic_proxy`,
  `bedrock`, `nv_build`, `nv_inference`, or a local CLI provider `claude_cli`,
  `codex_cli`, `gemini_cli`), used only when `use_llm` is true. Unknown values
  are rejected by SkillSpector's core at scan time and returned as a JSON error.
- `model`: optional model override, used only when `use_llm` is true.

## LLM-backed analysis

Static analysis is the default and needs no credentials. To enable the semantic
pass, configure provider credentials in the environment Hermes runs in (e.g.
`SKILLSPECTOR_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`), then call the
tool with `use_llm=true`:

```text
Use skillspector_scan on ./my-skill with use_llm=true and provider=anthropic.
```

These environment variables are declared as `optional_env` in `plugin.yaml`, so
Hermes surfaces them during `hermes plugins install` but does not disable the
plugin when they are absent.

## Handler contract

`skillspector_scan` always returns a JSON string and never raises. Failures —
including SkillSpector not being installed in the Hermes environment — come back
as `{"error": ...}`, per the Hermes handler contract.

## Remove

```bash
hermes plugins disable skillspector_hermes
rm -rf ~/.hermes/plugins/skillspector_hermes
```
