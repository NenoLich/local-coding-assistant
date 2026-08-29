# LOCCA development notes

Local Coding Assistant (LOCCA): a local-first AI coding assistant. Python 3.12, managed with `uv` (see `pyproject.toml`, `uv.lock`).

## Entrypoints & architecture

- Console script `locca` → `src/local_coding_assistant/cli/main.py` (Typer). Subcommands: `run`, `serve`, `tool`, `config`, `provider`, `sandbox`, `dashboard`.
- `core/bootstrap.py` wires everything up: `ConfigManager`/`EnvManager`/`PathManager` → `LLMService`, `ToolManager`, `RuntimeManager`, `SandboxManager`. CLI commands pull a fresh context from `bootstrap()`.
- **`agent/frame_agent.py` is the current agent implementation. `agent/agent_loop.py` and `agent/langgraph_agent.py` are deprecated — do not extend them.**
- `sandbox/` runs user code in Docker (`locca-sandbox:latest` image). `sandbox/guest/*` runs *inside* the container; `config/defaults.yaml` disables the sandbox by default (`sandbox.enabled: false`).

## Commands

```bash
uv run locca run query "..."          # CLI entrypoint; run --help to list flags
uvx ty check --ignore unresolved-import src/   # typecheck (whole src/, as pre-commit does; tests are not checked)
ruff check --fix .                    # lint
ruff format .                         # format
uv run task test-unit                 # pytest -v -s -x tests/unit
uv run task test-integration          # pytest -v -s -x tests/integration
uv run task test-e2e                  # pytest -v -s -x tests/e2e
uv run task test-all                  # pytest -v -s -x tests
uv run pytest tests/unit/<pkg>/test_x.py -vx   # single test
```

Gotchas:
- **Plain `pytest` (no args) collects `tests/` AND `benchmark/`** (`testpaths` in pyproject), which is slow. Always pass an explicit path.
- `uv run` auto-syncs from `uv.lock` first (needs network, can be slow). Use `uv run --no-sync ...` or `.venv/bin/...` for quick checks. If `.venv` imports fail with missing-module errors that shouldn't exist, rebuild with `uv sync`.
- `asyncio_mode = "auto"` in pyproject: async tests run automatically, no `@pytest.mark.asyncio` needed.
- Golden tests compare exact output against `tests/unit/cli/golden/*.txt` (pytest-golden); regenerate with `pytest ... --update-goldens`.
- Pre-commit (`.pre-commit-config.yaml`): lint/format/ty + unit tests run on commit, integration + e2e on push. Install with `pre-commit install`.

## Configuration & paths

- Env files load in order, later overrides earlier: `.env` → `.env.${LOCCA_ENV}` → `.env.local` (gitignored). Variables use `LOCCA_` prefix and can interpolate `${VAR}`. Root conftest forces `LOCCA_ENV=test` for all tests.
- Resolve filesystem paths through `PathManager.resolve()` (env-dependent; construct via `get_env_manager()`), never hardcode. Real aliases: `@config`, `@data`, `@target`, `@cache`, `@log`, `@module`, `@project`, `@tools`, `@templates`.
- **`@root` and `@logs` (documented in README) are NOT real aliases** — `path_manager.py` has no "root"/"logs" handlers, so those resolve to a literal `./@root/...` path under the project root. Use `@project` / `@log`.
- Config is three-layer: global → session → call (see `config/defaults.yaml`, `config_manager.py`). Local overrides go in `config/*.local.yaml` and `.env.local`, both gitignored.
- Real API keys live in `.env.local` / `config/*.local.yaml` (gitignored) — never commit them.

## Conventions

- Type hints on all signatures. Data structures are Pydantic models or dataclasses; pass typed contracts between modules, not dicts. Use f-strings and PEP 604 unions (`str | None`).
- Errors: raise custom exceptions from `core/exceptions.py` (or module-local `exceptions.py`); CLI commands must be wrapped with `safe_entrypoint` from `core/error_handler.py`.
- Logging: `utils/logging.get_logger("a.b.c")` (structlog). Pass structured data as keyword args, e.g. `logger.info("msg", data=thing)`, not string interpolation.
- All I/O is async.
- Ruff (pyproject): max-complexity 10; `tests/**` is allowed `assert`/`print`; **`**/sandbox/guest/tools_api.py` is excluded from ruff + pyupgrade** because it runs in the trimmed sandbox runtime — leave it untouched.

## Adding tools & providers

- Regular tools: register in `config/tools.default.yaml`, implement a class with `run()` (+ optional `stream()`); inputs are JSON-schema validated. Sandbox tools: register in `config/sandbox_tools.yaml`, args come from the docstring.
- Providers are declarative: `config/providers.default.yaml` are built-ins, `locca provider add ...` writes to `config/providers.local.yaml`. `config/` is the source of truth; `README.md` mirrors this but lags in places (see the `@root`/`@logs` note above).