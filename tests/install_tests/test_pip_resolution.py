import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from installer.runtime_matrix import (
    resolve_runtime_spec,
    resolve_torch_install,
    torch_index_url,
)


@dataclass(frozen=True, slots=True)
class _ResolverCase:
    name: str
    blender_version: tuple[int, int, int]
    python_version: tuple[int, int]
    abi: str
    channel: str
    expected_numpy_prefix: str
    expected_torch_prefix: str
    expected_torch_contains: str
    expected_opencv_version: str
    expected_diffusers_version: str
    expected_transformers_version: str
    expected_accelerate_version: str
    expected_safetensors_version: str
    expected_peft_version: str


def _resolver_cases() -> list[_ResolverCase]:
    cases = [
        _ResolverCase(
            name="win-blender50-cpu",
            blender_version=(5, 0, 0),
            python_version=(3, 11),
            abi="cp311",
            channel="cpu",
            expected_numpy_prefix="1.",
            expected_torch_prefix="2.8.0+cpu",
            expected_torch_contains="+cpu",
            expected_opencv_version="4.11.0.86",
            expected_diffusers_version="0.38.0",
            expected_transformers_version="5.8.1",
            expected_accelerate_version="1.13.0",
            expected_safetensors_version="0.8.0rc0",
            expected_peft_version="0.19.1",
        ),
        _ResolverCase(
            name="win-blender51-cpu",
            blender_version=(5, 1, 0),
            python_version=(3, 13),
            abi="cp313",
            channel="cpu",
            expected_numpy_prefix="2.3.",
            expected_torch_prefix="2.",
            expected_torch_contains="+cpu",
            expected_opencv_version="4.13.0.92",
            expected_diffusers_version="0.38.0",
            expected_transformers_version="5.8.1",
            expected_accelerate_version="1.13.0",
            expected_safetensors_version="0.8.0rc0",
            expected_peft_version="0.19.1",
        ),
    ]
    if os.getenv("DIFFUSEDTEXTURE_FULL_RESOLUTION_MATRIX") == "1":
        cases.extend(
            [
                _ResolverCase(
                    name="win-blender50-cu129",
                    blender_version=(5, 0, 0),
                    python_version=(3, 11),
                    abi="cp311",
                    channel="cu129",
                    expected_numpy_prefix="1.",
                    expected_torch_prefix="2.8.0+cu129",
                    expected_torch_contains="+cu129",
                    expected_opencv_version="4.11.0.86",
                    expected_diffusers_version="0.38.0",
                    expected_transformers_version="5.8.1",
                    expected_accelerate_version="1.13.0",
                    expected_safetensors_version="0.8.0rc0",
                    expected_peft_version="0.19.1",
                ),
                _ResolverCase(
                    name="win-blender51-cu130",
                    blender_version=(5, 1, 0),
                    python_version=(3, 13),
                    abi="cp313",
                    channel="cu130",
                    expected_numpy_prefix="2.3.",
                    expected_torch_prefix="2.",
                    expected_torch_contains="+cu130",
                    expected_opencv_version="4.13.0.92",
                    expected_diffusers_version="0.38.0",
                    expected_transformers_version="5.8.1",
                    expected_accelerate_version="1.13.0",
                    expected_safetensors_version="0.8.0rc0",
                    expected_peft_version="0.19.1",
                ),
            ]
        )
    return cases


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


def _run_resolution(case: _ResolverCase, report_path: Path) -> dict:
    runtime_spec = resolve_runtime_spec(
        blender_version=case.blender_version,
        python_version=case.python_version,
    )
    install_channel, torch_requirement, _ = resolve_torch_install(
        case.channel,
        platform="win32",
        blender_version=case.blender_version,
    )
    index_url, _ = torch_index_url(install_channel)

    command = [
        *_pip_base_command(),
        "install",
        "--dry-run",
        "--ignore-installed",
        "--only-binary=:all:",
        "--report",
        str(report_path),
        "--index-url",
        "https://pypi.org/simple",
        "--extra-index-url",
        index_url,
        "--platform",
        "win_amd64",
        "--python-version",
        f"{case.python_version[0]}.{case.python_version[1]}",
        "--implementation",
        "cp",
        "--abi",
        case.abi,
        torch_requirement,
        *runtime_spec.runtime_requirements,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(report_path.read_text(encoding="utf-8"))


def _versions_by_name(report: dict) -> dict[str, str]:
    install_entries = report.get("install", [])
    versions: dict[str, str] = {}
    for item in install_entries:
        metadata = item.get("metadata", {})
        name = str(metadata.get("name", "")).lower()
        version = str(metadata.get("version", ""))
        if name and version:
            versions[name] = version
    return versions


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    ("case"),
    _resolver_cases(),
    ids=lambda case: case.name,
)
def test_runtime_dependencies_resolve_for_supported_windows_matrix(
    case: _ResolverCase,
    tmp_path: Path,
) -> None:
    report = _run_resolution(case, tmp_path / f"{case.name}.json")
    versions = _versions_by_name(report)

    assert versions["numpy"].startswith(case.expected_numpy_prefix)
    assert versions["torch"].startswith(case.expected_torch_prefix)
    assert case.expected_torch_contains in versions["torch"]
    assert versions["opencv-python-headless"] == case.expected_opencv_version
    assert versions["diffusers"] == case.expected_diffusers_version
    assert versions["transformers"] == case.expected_transformers_version
    assert versions["accelerate"] == case.expected_accelerate_version
    assert versions["safetensors"] == case.expected_safetensors_version
    assert versions["peft"] == case.expected_peft_version
