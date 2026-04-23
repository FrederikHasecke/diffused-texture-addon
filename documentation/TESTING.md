# Testing

## Default Local Checks

Run these from the repository root for normal implementation work:

1. `uvx ruff check . --config .\pyproject.toml`
2. `uvx ty check . --python-version 3.13`
3. `uv run pytest tests/unit_tests tests/integration_tests -m "not slow and not gpu and not network and not e2e" -q`

The matching VS Code tasks are `check-fast` and `check-installer`.

The fast `uv run pytest ...` suite now targets the default local Python 3.13 + Blender 5.1 development environment created by `uv sync`. The shared `uv` configuration deliberately overrides the published `bpy` NumPy cap so the local dev environment matches the CY2026 NumPy 2.3 lane. Blender `<5.1` and Python `3.11` compatibility remain covered by the installer matrix tests and later by Blender smoke tests against real Blender binaries.

## Installer Matrix Tests

The installer matrix is defined in `installer/runtime_matrix.py` and the pinned runtime requirement files in `installer/constraints/`.

Use these tests when installer or dependency logic changes:

1. `uv run pytest tests/unit_tests/test_runtime_matrix.py -q`
2. `uv run pytest tests/install_tests/test_pip_resolution.py -q`

`test_pip_resolution.py` uses `pip --dry-run --report` so it can validate both Windows dependency lanes without performing a full install: Blender `<5.1` on `py311` and Blender `5.1+` on `py313`.

By default the resolver test covers the CPU boundary cases. To include the slower CUDA resolver cases as well:

1. `$env:DIFFUSEDTEXTURE_FULL_RESOLUTION_MATRIX = "1"`
2. `uv run pytest tests/install_tests/test_pip_resolution.py -q`

## Real Install Smoke

To perform an actual install into a temporary target directory for the current local Python runtime:

1. `$env:DIFFUSEDTEXTURE_INSTALL_SMOKE = "1"`
2. `uv run pytest tests/install_tests/test_real_install.py -q`

This is intentionally opt-in because it downloads and installs the full runtime dependency set.

## Blender Smoke And E2E

The existing Blender integration and E2E tests remain useful, but they are slower and should stay manual or scheduled while both dependency lanes stay supported.
