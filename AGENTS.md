# Repository Guidelines

## Project Structure & Module Organization

SkillSpector is a Python 3.12+ security scanner for AI agent skills. Production
code lives in `src/skillspector/`: the CLI is `cli.py`, the LangGraph pipeline
is in `graph.py` and `nodes/`, and analyzer implementations belong in
`nodes/analyzers/`. LLM providers are isolated in `providers/`; bundled YARA
rules live in `yara_rules/`. Keep analyzer-specific fixtures under
`tests/fixtures/` and mirror source areas in `tests/unit/`, `tests/nodes/`,
`tests/integration/`, or `tests/provider/`. Project design and developer notes
are in `docs/`.

## Build, Test, and Development Commands

Create and activate a virtual environment first, then run `make install-dev`
(uses `uv` when available, otherwise pip). Useful commands:

- `make test-unit` — run the default offline unit suite.
- `make test` — run unit and integration tests; integration tests may invoke an LLM.
- `make test-cov` — run default tests and report branch coverage.
- `make lint` / `make format-check` — check Ruff linting and formatting.
- `make format` — apply Ruff fixes and formatting.
- `make build` — build the Python distribution; `make docker-smoke` builds and smoke-tests the image.

Run `make test-provider openai` (or `anthropic`, `nv_build`) only with the
corresponding credentials configured.

## Coding Style & Naming Conventions

Use four-space Python indentation, type annotations for all function
definitions, and a 100-character Ruff line length. Ruff enforces errors,
imports, naming, modern Python, bugbear, and comprehension rules; use
`make format` before submission. Use `snake_case` for modules, functions, and
variables; `PascalCase` for classes; and descriptive analyzer module names such
as `static_patterns_tool_misuse.py`. Include the repository SPDX license header
in every new source file.

## Testing Guidelines

Write focused pytest tests named `test_<behavior>.py` or `test_<behavior>`.
Add representative safe and malicious fixture skills whenever an analyzer or
detection rule changes. Unit tests must not require external services. Mark
full-graph tests `integration` and live endpoint tests `provider`; the default
pytest configuration excludes both.

## Commits & Pull Requests

Recent history uses concise conventional-style subjects such as
`feat:`, `fix:`, `test:`, `docs:`, and `chore:`; keep each commit scoped to one
intent. Every commit requires a DCO sign-off—use `git commit -s`. Open or link
an issue before the PR, explain the change and validation performed, and add
tests and fixtures for analyzer changes. Include screenshots or sample output
when a user-facing report or CLI behavior changes.
