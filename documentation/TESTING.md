# Testing

All commands run from the repository root. `uv sync` creates the default development environment: Python 3.13 with the Blender 5.2 LTS `bpy` wheel.

Environment variables are shown in POSIX shell syntax. In PowerShell, use `$env:NAME = "1"` instead of `NAME=1`.

## Default Local Checks

Run these for normal implementation work:

1. `uv run ruff check . --config pyproject.toml`
2. `uvx ty check . --python-version 3.13`
3. `uv run pytest tests/unit_tests tests/integration_tests -m "not slow and not gpu and not network and not e2e" -q`

Blender 5.2's `bpy` wheel supports the NumPy 2.3 lane natively, so no dependency override is needed. Compatibility with Blender `<5.1` and Python 3.11 is covered by the installer matrix tests and the built artifact smoke test.

## Installer Matrix Tests

The installer matrix is defined in `installer/runtime_matrix.py` and the pinned requirement files in `installer/constraints/`. Run these when installer or dependency logic changes:

1. `uv run pytest tests/unit_tests/test_runtime_matrix.py -q`
2. `uv run pytest tests/install_tests/test_pip_resolution.py -q`

`test_pip_resolution.py` uses `pip install --dry-run --report`, so it validates both Windows dependency lanes without a full install: Blender `<5.1` on Python 3.11 and Blender 5.2 on Python 3.13.

By default the resolver test covers the CPU boundary cases. To also resolve representative CUDA dependency graphs and verify hash-backed wheels for every selectable CUDA channel plus the Linux ROCm 6.3 lane:

```bash
DIFFUSEDTEXTURE_FULL_RESOLUTION_MATRIX=1 uv run pytest tests/install_tests/test_pip_resolution.py -q
```

## Real Install Smoke

Performs an actual install of the full runtime dependency set into a temporary directory for the current Python. Opt-in because it downloads several GB:

```bash
DIFFUSEDTEXTURE_INSTALL_SMOKE=1 uv run pytest tests/install_tests/test_real_install.py -q
```

## Built Artifact Smoke

`.github/workflows/artifact-smoke.yml` builds the release ZIP and tests it on GitHub runners for both Blender/Python lanes (`bpy` 5.0.1 on Python 3.11, `bpy` 5.2.0 on Python 3.13). It checks the archive root, compiles every packaged Python file, imports the extracted add-on outside the checkout, and requires a clean register/unregister cycle.

The smoke installs only `bpy`, NumPy, Pillow, OpenCV and dotenv; it never downloads PyTorch, diffusion packages or model weights. To check a locally built ZIP:

```bash
python scripts/ci/smoke_built_addon.py path/to/diffused_texture_addon.zip
```

## End-to-End Tests

`tests/e2e_tests/test_run_parallel.py` runs a real generation for each operation mode on a Suzanne mesh, using 1 inference step and a tiny SD resolution. It needs a GPU, network access and the default SD 1.5 models, and stays manual:

```bash
uv run pytest tests/e2e_tests -m e2e -q
```

- Set `HF_HOME` (or put it in a `.env` file) to reuse an existing Hugging Face cache.
- `DIFFUSEDTEXTURE_E2E_TIMEOUT` sets the per-test timeout in seconds (default `180`). The first run downloads the models and needs much longer.
