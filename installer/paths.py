import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import bpy


def ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except Exception:  # noqa: BLE001
        import ensurepip

        ensurepip.bootstrap()
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=False,
    )


def deps_target_dir() -> Path:
    base = Path(bpy.utils.user_resource("SCRIPTS", path="", create=True))
    target = base / "modules" / "diffusedtexture_deps"
    target.mkdir(parents=True, exist_ok=True)
    return target


def run(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(  # noqa: S603
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return p.returncode, p.stdout  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        return 1, f"<exec failed: {e!s}>"


def make_importable(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    importlib.invalidate_caches()


def clean_pip_env() -> dict[str, str]:
    """Scrub PIP_* vars and disable user site to keep installs deterministic."""
    env = os.environ.copy()
    for k in list(env.keys()):
        if k.upper().startswith("PIP_"):
            env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def read_pyproject_runtime_deps(pyproject_path: Path) -> list[str]:
    """Return [project].dependencies from pyproject.toml as list of requirement strings.

    Filters out 'torch' and 'bpy' because the operator handles those specially.
    """
    if not pyproject_path.exists():
        return []

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    deps = list(data.get("project", {}).get("dependencies", []) or [])
    cleaned: list[str] = []
    for d in deps:
        s = str(d).strip()
        # Avoid letting this pass through the generic install step
        lower = s.lower()
        if lower.startswith("torch"):  # torch, torch==x, torch>=x, torch~=x
            continue
        if lower.startswith("bpy"):
            continue
        cleaned.append(s)
    return cleaned
