import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
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
    runtime_platform: str
    pip_platforms: tuple[str, ...]
    channel: str
    expected_numpy_prefix: str
    expected_torch_prefix: str
    expected_torch_contains: str


@dataclass(frozen=True, slots=True)
class _PublishedWheelCase:
    channel: str
    version: str
    python_tag: str
    platform_tag: str


def _resolver_cases() -> list[_ResolverCase]:
    cases = [
        _ResolverCase(
            name="win-blender50-cpu",
            blender_version=(5, 0, 0),
            python_version=(3, 11),
            abi="cp311",
            runtime_platform="win32",
            pip_platforms=("win_amd64",),
            channel="cpu",
            expected_numpy_prefix="1.",
            expected_torch_prefix="2.8.0+cpu",
            expected_torch_contains="+cpu",
        ),
        _ResolverCase(
            name="win-blender52-cpu",
            blender_version=(5, 2, 0),
            python_version=(3, 13),
            abi="cp313",
            runtime_platform="win32",
            pip_platforms=("win_amd64",),
            channel="cpu",
            expected_numpy_prefix="2.3.",
            expected_torch_prefix="2.12.0+cpu",
            expected_torch_contains="+cpu",
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
                    runtime_platform="win32",
                    pip_platforms=("win_amd64",),
                    channel="cu129",
                    expected_numpy_prefix="1.",
                    expected_torch_prefix="2.8.0+cu129",
                    expected_torch_contains="+cu129",
                ),
                _ResolverCase(
                    name="win-blender52-cu130",
                    blender_version=(5, 2, 0),
                    python_version=(3, 13),
                    abi="cp313",
                    runtime_platform="win32",
                    pip_platforms=("win_amd64",),
                    channel="cu130",
                    expected_numpy_prefix="2.3.",
                    expected_torch_prefix="2.12.0+cu130",
                    expected_torch_contains="+cu130",
                ),
            ]
        )
    return cases


def _published_wheel_cases() -> list[_PublishedWheelCase]:
    return [
        _PublishedWheelCase("cu126", "2.8.0", "cp311", "win_amd64"),
        _PublishedWheelCase("cu128", "2.8.0", "cp311", "win_amd64"),
        _PublishedWheelCase("cu129", "2.8.0", "cp311", "win_amd64"),
        _PublishedWheelCase("cpu", "2.12.0", "cp313", "win_amd64"),
        _PublishedWheelCase("cu126", "2.9.1", "cp313", "win_amd64"),
        _PublishedWheelCase("cu128", "2.9.1", "cp313", "win_amd64"),
        _PublishedWheelCase("cu129", "2.9.0", "cp313", "win_amd64"),
        _PublishedWheelCase("cu130", "2.12.0", "cp313", "win_amd64"),
        _PublishedWheelCase(
            "rocm6.3",
            "2.9.0",
            "cp313",
            "manylinux_2_28_x86_64",
        ),
    ]


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
        platform=case.runtime_platform,
        blender_version=case.blender_version,
    )
    index_url, _ = torch_index_url(install_channel)
    platform_args = [
        argument
        for platform_name in case.pip_platforms
        for argument in ("--platform", platform_name)
    ]

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
        *platform_args,
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
def test_runtime_dependencies_resolve_for_supported_matrix(
    case: _ResolverCase,
    tmp_path: Path,
) -> None:
    report = _run_resolution(case, tmp_path / f"{case.name}.json")
    versions = _versions_by_name(report)
    runtime_spec = resolve_runtime_spec(
        blender_version=case.blender_version,
        python_version=case.python_version,
    )

    assert versions["numpy"].startswith(case.expected_numpy_prefix)
    assert versions["torch"].startswith(case.expected_torch_prefix)
    assert case.expected_torch_contains in versions["torch"]

    for requirement_text in runtime_spec.runtime_requirements:
        requirement = Requirement(requirement_text)
        package_name = requirement.name.lower()
        assert package_name in versions
        assert Version(versions[package_name]) in requirement.specifier


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    ("case"),
    _published_wheel_cases(),
    ids=lambda case: (
        f"{case.channel}-{case.version}-{case.python_tag}-{case.platform_tag}"
    ),
)
def test_accelerator_baseline_wheels_are_hash_published(
    case: _PublishedWheelCase,
) -> None:
    if os.getenv("DIFFUSEDTEXTURE_FULL_RESOLUTION_MATRIX") != "1":
        pytest.skip("Set DIFFUSEDTEXTURE_FULL_RESOLUTION_MATRIX=1 for all channels")

    index_url, _ = torch_index_url(case.channel)
    with urllib.request.urlopen(f"{index_url}/torch/", timeout=30) as response:
        index_html = response.read().decode("utf-8")

    encoded_filename = (
        f"torch-{case.version}%2B{case.channel}-{case.python_tag}-"
        f"{case.python_tag}-{case.platform_tag}.whl"
    )
    hash_backed_link = re.compile(
        rf'href="[^"]*/{re.escape(encoded_filename)}#sha256=[0-9a-f]{{64}}"',
    )
    assert hash_backed_link.search(index_html), (
        f"No hash-backed wheel link found for {encoded_filename} in {index_url}"
    )
