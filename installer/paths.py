import importlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import bpy

try:
    from ..diagnostics import get_logger
except ImportError:
    from diagnostics import get_logger

_ACTIVE_ENV_FILE = ".active"
_logger = get_logger("installer.paths")


def ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except Exception:  # noqa: BLE001
        import ensurepip

        ensurepip.bootstrap()


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


def _format_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def _log_subprocess_output(label: str, output: str) -> None:
    for line in output.splitlines():
        _logger.debug("[%s] %s", label, line)


def run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    *,
    label: str | None = None,
) -> tuple[int, str]:
    command_label = label or Path(cmd[0]).name
    started_at = time.monotonic()
    _logger.debug("Starting subprocess [%s]: %s", command_label, _format_command(cmd))
    try:
        p = subprocess.run(  # noqa: S603
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        duration = time.monotonic() - started_at
        _log_subprocess_output(command_label, p.stdout or "")
        _logger.debug(
            "Finished subprocess [%s] rc=%s duration=%.2fs",
            command_label,
            p.returncode,
            duration,
        )
        return p.returncode, p.stdout  # noqa: TRY300
    except Exception as e:
        _logger.exception(
            "Subprocess launch failed [%s]: %s",
            command_label,
            _format_command(cmd),
        )
        return 1, f"<exec failed: {e!s}>"


def run_stream(
    cmd: list[str],
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    *,
    label: str | None = None,
) -> tuple[int, str]:
    output_lines: list[str] = []
    command_label = label or Path(cmd[0]).name
    started_at = time.monotonic()
    _logger.debug("Starting subprocess [%s]: %s", command_label, _format_command(cmd))
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
                _logger.debug("[%s] %s", command_label, clean)
                if on_line is not None:
                    on_line(clean)
        rc = proc.wait()
        duration = time.monotonic() - started_at
        _logger.debug(
            "Finished subprocess [%s] rc=%s duration=%.2fs",
            command_label,
            rc,
            duration,
        )
        return rc, "\n".join(output_lines)
    except Exception as e:
        _logger.exception(
            "Streaming subprocess launch failed [%s]: %s",
            command_label,
            _format_command(cmd),
        )
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
