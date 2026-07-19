import os
import subprocess
import sys
from pathlib import Path

import pytest

from installer.runtime_matrix import (
    resolve_runtime_spec,
    resolve_torch_install,
    torch_index_url,
)


def _pip_base_command() -> list[str]:
    pip_version = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_version.returncode != 0:
        bootstrap = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert bootstrap.returncode == 0, bootstrap.stdout + bootstrap.stderr
    return [sys.executable, "-m", "pip"]


def _current_blender_boundary() -> tuple[int, int, int]:
    if sys.version_info[:2] == (3, 11):
        return (5, 0, 0)
    if sys.version_info[:2] == (3, 13):
        return (5, 2, 0)
    msg = (
        "Real install smoke test only supports the current local dev lanes: Python 3.13 for Blender 5.1+ or Python 3.11 for Blender <5.1, "
        f"got {sys.version_info[0]}.{sys.version_info[1]}."
    )
    raise RuntimeError(msg)


@pytest.mark.network
@pytest.mark.slow
def test_real_install_smoke_current_python(tmp_path: Path) -> None:
    if os.getenv("DIFFUSEDTEXTURE_INSTALL_SMOKE") != "1":
        pytest.skip(
            "Set DIFFUSEDTEXTURE_INSTALL_SMOKE=1 to run real install smoke tests."
        )

    blender_version = _current_blender_boundary()
    runtime_spec = resolve_runtime_spec(
        blender_version=blender_version,
        python_version=sys.version_info[:2],
    )
    install_channel, torch_requirement, _ = resolve_torch_install(
        "cpu",
        platform=sys.platform,
        blender_version=blender_version,
    )
    index_url, _ = torch_index_url(install_channel)
    target = tmp_path / "deps"

    install_command = [
        *_pip_base_command(),
        "install",
        "--no-cache-dir",
        "--upgrade",
        "--prefer-binary",
        "--only-binary=:all:",
        "--target",
        str(target),
        "--index-url",
        "https://pypi.org/simple",
        "--extra-index-url",
        index_url,
        torch_requirement,
        *runtime_spec.runtime_requirements,
    ]
    completed = subprocess.run(
        install_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    script = (
        f"import site; site.addsitedir(r'{target}');"
        "import accelerate, cv2, diffusers, numpy, peft, PIL, safetensors, torch, transformers"
    )
    import_check = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stdout + import_check.stderr
