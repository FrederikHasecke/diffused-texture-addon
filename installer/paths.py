import importlib
import os
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import bpy

_ACTIVE_ENV_FILE = ".active"


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


def deps_root_dir() -> Path:
    base = Path(bpy.utils.user_resource("SCRIPTS", path="", create=True))
    target = base / "modules" / "diffusedtexture_deps"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _latest_env_dir(root: Path) -> Path | None:
    env_dirs = [p for p in root.glob("env_*") if p.is_dir()]
    if not env_dirs:
        return None
    return max(env_dirs, key=lambda p: p.stat().st_mtime)


def _has_legacy_flat_install(root: Path) -> bool:
    markers = ("torch", "diffusers", "PIL", "cv2", "accelerate")
    return any((root / name).exists() for name in markers)


def deps_target_dir() -> Path:
    """Return the active dependency directory.

    Legacy installs used the root directory directly. New installs use
    side-by-side environment directories under the same root and track the
    active one via a marker file.
    """
    root = deps_root_dir()
    marker = root / _ACTIVE_ENV_FILE
    if marker.exists():
        try:
            active_name = marker.read_text(encoding="utf-8").strip()
        except OSError:
            active_name = ""
        if active_name:
            candidate = root / active_name
            if candidate.is_dir():
                return candidate

    if _has_legacy_flat_install(root):
        return root

    latest = _latest_env_dir(root)
    if latest is not None:
        return latest
    return root


def new_deps_target_dir() -> Path:
    root = deps_root_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = root / f"env_{stamp}_{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def set_active_deps_target(path: Path) -> None:
    root = deps_root_dir().resolve()
    candidate = path.resolve()
    if candidate.parent != root:
        msg = f"Active deps target must be a direct child of {root}"
        raise ValueError(msg)

    tmp = root / f"{_ACTIVE_ENV_FILE}.tmp"
    marker = root / _ACTIVE_ENV_FILE
    tmp.write_text(candidate.name, encoding="utf-8")
    tmp.replace(marker)


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


def run_stream(
    cmd: list[str],
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    output_lines: list[str] = []
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                clean = line.rstrip("\n")
                output_lines.append(clean)
                if on_line is not None:
                    on_line(clean)
        rc = proc.wait()
        return rc, "\n".join(output_lines)
    except Exception as e:  # noqa: BLE001
        return 1, f"<exec failed: {e!s}>"


def make_importable(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
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

    if not deps:
        msg = "No dependencies found in pyproject.toml"
        raise ValueError(msg)

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
