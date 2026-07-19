# Agent Notes

Use the repository root as the working directory for all commands.

## Default Checks

Run these after every implementation unless the user explicitly asks otherwise:

- `uvx ruff check . --config .\pyproject.toml`
- `uvx ty check . --python-version 3.13`
- `uv run pytest tests/unit_tests tests/integration_tests -m "not slow and not gpu and not network and not e2e" -q`

The fast `uv run pytest ...` suite assumes the default local Python 3.13 + Blender 5.2 LTS development environment created by `uv sync`. Blender 5.2's `bpy` wheel natively supports the CY2026 NumPy 2.3 lane, so no dependency override is required. Cross-version dependency coverage for the legacy `py311` lane is handled by the installer matrix tests below.

## Installer Changes

If you change dependency installation, runtime version logic, `pyproject.toml`, or CI for installs, also run:

- `uv run pytest tests/unit_tests/test_runtime_matrix.py -q`
- `uv run pytest tests/install_tests/test_pip_resolution.py -q`

Set `DIFFUSEDTEXTURE_FULL_RESOLUTION_MATRIX=1` before `test_pip_resolution.py` when you want representative CUDA dependency resolution plus hash-backed wheel checks for every selectable CUDA channel and the Linux ROCm 6.3 lane.

## Opt-In Heavy Checks

- Real install smoke test: `DIFFUSEDTEXTURE_INSTALL_SMOKE=1 uv run pytest tests/install_tests/test_real_install.py -q`
- Existing Blender E2E tests remain opt-in because they are slow and may download models.

## Release And Commit Policy

- `master` is the only release branch.
- `release-please` owns normal release tags, changelog entries, and version bumps.
- Do not manually edit `pyproject.toml` or `blender_manifest.toml` only to cut a release. Those files are updated by the release PR.
- When the user explicitly asks for a commit, use Conventional Commits so the release automation can derive semver correctly.
- Semver mapping for commits:
  - `feat:` = minor release for a backward-compatible feature.
  - `fix:` = patch release for a bug fix or broken runtime/install behavior.
  - `type!:` or `BREAKING CHANGE:` = major release for incompatible changes.
  - `docs:`, `test:`, `chore:`, `refactor:`, `ci:`, `build:`, and `style:` = no release by default unless marked breaking.
- Repo-specific guidance:
  - Dropping Blender or Python support is major.
  - Adding a new addon capability, model family, or user-facing workflow is minor.
  - Correcting broken installer behavior, packaging, or runtime handling is patch.
- Prefer squash/rebase-friendly Conventional Commit messages. Non-conventional merge commit messages will not drive release-please bumps.

## Intent

The runtime dependency matrix is defined in `installer/runtime_matrix.py` and the pinned requirement files under `installer/constraints/`. Update those first, then update tests.
