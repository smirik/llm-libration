# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `llm_libration/` with submodules for CLI (`cli/` and `cli.py`), provider config (`config.py`, `llm/`), analysis logic (`analyzer.py`, `types.py`), and assets (`data/`, `benchmark/`).
- Tests reside in `tests/` mirroring modules (e.g., `test_analyzer.py`, `cli/` smoke tests). Add new tests beside the code they cover.
- Utility scripts live in `scripts/` (e.g., `patch_transformers.py` for MLX compatibility). Design notes are under `docs/`.

## Setup, Build, and Development Commands
- `make install` — create `.venv` and install `.[dev]`.
- `source .venv/bin/activate` — activate the environment before hacking.
- `make test` / `make test-verbose` — run the pytest suite; `make coverage` adds HTML/terminal coverage.
- `make format` / `make lint` / `make check` — apply Black, run Flake8, or do all checks.
- `make build` — build wheel/sdist; `make clean` or `make clean-all` removes artifacts/venv.
- CLI examples: `llm-libration run input/demo.png --provider openai`, `llm-libration plot input/463.csv`, `llm-libration benchmark input/benchmark --simplified`.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indent; prefer type hints for public surfaces.
- Black with `line-length = 140` and `skip-string-normalization = true`; run `make format`.
- Flake8 ignores `E203`/`W503`; keep lines ≤140 chars otherwise.
- Modules and functions use `snake_case`; classes in `PascalCase`; constants upper snake.
- Keep CLI flags descriptive and aligned with existing `click` patterns.

## Testing Guidelines
- Pytest config in `pyproject.toml` enforces markers/config strictness and coverage (`--cov=llm_libration`).
- Name tests `test_*` and group by feature (unit tests per module; integration via CLI fixtures in `tests/cli/`).
- Provide minimal fixtures and sample inputs in `tests/data` or new module-local fixtures; avoid large assets in git.
- Aim to maintain or improve coverage; add regression tests for reported issues.

## Commit & Pull Request Guidelines
- Commits follow Conventional Commit style seen in history (`feat:`, `docs:`, `fix:`, etc.); keep scope small and message imperative.
- PRs should describe intent, key changes, and testing performed (`make test`, `make check`, CLI sanity commands).
- Link issues or benchmarks when relevant; include before/after notes or screenshots for output changes.
- Avoid committing secrets; use `.env.dist` as a template and keep provider keys only in local `.env`.

## Configuration & Provider Tips
- Copy `.env.dist` to `.env`; set `LLM_PROVIDER` and API keys. Do not commit `.env`.
- For MLX on macOS, run `python scripts/patch_transformers.py` after dependency sync until upstream fix is released.
- Local Ollama requires `ollama serve` and model pulls (e.g., `ollama pull qwen2.5vl:7b`); keep prompts in env vars (`PROMPT_TEMPLATE`, `PROMPT_TEMPLATE_SIMPLIFIED`).
