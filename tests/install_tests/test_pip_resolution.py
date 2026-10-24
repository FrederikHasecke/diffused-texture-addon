import json
import os
import re
import shutil
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
    runtime_platform: str
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
            runtime_platform="win32",
            channel="cpu",
            expected_numpy_prefix="1.",
            expected_torch_prefix="2.8.0+cpu",
            expected_torch_contains="+cpu",
        ),
        _ResolverCase(
            name="win-blender52-cpu",
            blender_version=(5, 2, 0),
            python_version=(3, 13),
            runtime_platform="win32",
            channel="cpu",
            expected_numpy_prefix="2.3.",
            expected_torch_prefix="2.14.1+cpu",
            expected_torch_contains="+cpu",
        ),
        _ResolverCase(
            name="linux-blender52-cpu",
            blender_version=(5, 2, 0),
            python_version=(3, 13),
            runtime_platform="linux",
            channel="cpu",
            expected_numpy_prefix="2.3.",
            expected_torch_prefix="2.14.1+cpu",
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
                    runtime_platform="win32",
                    channel="cu129",
                    expected_numpy_prefix="1.",
                    expected_torch_prefix="2.8.0+cu129",
                    expected_torch_contains="+cu129",
                ),
                _ResolverCase(
                    name="win-blender52-cu126",
                    blender_version=(5, 2, 0),
                    python_version=(3, 13),
                    runtime_platform="win32",
                    channel="cu126",
                    expected_numpy_prefix="2.3.",
                    expected_torch_prefix="2.14.1+cu126",
                    expected_torch_contains="+cu126",
                ),
                _ResolverCase(
                    name="win-blender52-cu128",
                    blender_version=(5, 2, 0),
                    python_version=(3, 13),
                    runtime_platform="win32",
                    channel="cu128",
                    expected_numpy_prefix="2.3.",
                    expected_torch_prefix="2.9.1+cu128",
                    expected_torch_contains="+cu128",
                ),
                _ResolverCase(
                    name="win-blender52-cu129",
                    blender_version=(5, 2, 0),
                    python_version=(3, 13),
                    runtime_platform="win32",
                    channel="cu129",
                    expected_numpy_prefix="2.3.",
                    expected_torch_prefix="2.9.0+cu129",
                    expected_torch_contains="+cu129",
                ),
                _ResolverCase(
                    name="win-blender52-cu130",
                    blender_version=(5, 2, 0),
                    python_version=(3, 13),
                    runtime_platform="win32",
                    channel="cu130",
                    expected_numpy_prefix="2.3.",
                    expected_torch_prefix="2.14.1+cu130",
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
        _PublishedWheelCase("cpu", "2.14.1", "cp313", "win_amd64"),
        _PublishedWheelCase("cu126", "2.14.1", "cp313", "win_amd64"),
        _PublishedWheelCase("cu128", "2.9.1", "cp313", "win_amd64"),
        _PublishedWheelCase("cu129", "2.9.0", "cp313", "win_amd64"),
        _PublishedWheelCase("cu130", "2.14.1", "cp313", "win_amd64"),
        _PublishedWheelCase("cpu", "2.14.1", "cp313", "manylinux_2_28_x86_64"),
        _PublishedWheelCase("cu126", "2.14.1", "cp313", "manylinux_2_28_x86_64"),
        _PublishedWheelCase("cu130", "2.14.1", "cp313", "manylinux_2_28_x86_64"),
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


_UV_PYTHON_PLATFORMS = {
    "linux": "x86_64-manylinux_2_28",
    "win32": "x86_64-pc-windows-msvc",
}


def _resolution_inputs(case: _ResolverCase) -> tuple[str, list[str]]:
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
    return index_url, [torch_requirement, *runtime_spec.runtime_requirements]


def _is_native_case(case: _ResolverCase) -> bool:
    return (
        case.runtime_platform == sys.platform
        and case.python_version == sys.version_info[:2]
    )


def _run_pip_resolution(case: _ResolverCase, tmp_path: Path) -> dict[str, str]:
    index_url, requirements = _resolution_inputs(case)
    report_path = tmp_path / f"{case.name}.json"
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
        *requirements,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return _versions_by_name(json.loads(report_path.read_text(encoding="utf-8")))


def _run_uv_resolution(case: _ResolverCase, tmp_path: Path) -> dict[str, str]:
    # pip --platform only changes wheel tags and still evaluates dependency
    # markers for the host, e.g. torch's Linux-only CUDA requirements. uv
    # evaluates markers for the target platform and Python version instead.
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to resolve non-host platforms")

    index_url, requirements = _resolution_inputs(case)
    requirements_path = tmp_path / f"{case.name}.in"
    requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    command = [
        uv,
        "pip",
        "compile",
        str(requirements_path),
        "--no-config",
        "--quiet",
        "--no-header",
        "--no-annotate",
        "--only-binary",
        ":all:",
        "--index-url",
        "https://pypi.org/simple",
        "--extra-index-url",
        index_url,
        # Match pip, which considers every index for every package.
        "--index-strategy",
        "unsafe-best-match",
        "--python-platform",
        _UV_PYTHON_PLATFORMS[case.runtime_platform],
        "--python-version",
        f"{case.python_version[0]}.{case.python_version[1]}",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return {
        match.group(1).lower(): match.group(2)
        for line in completed.stdout.splitlines()
        if (match := re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line))
    }


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
    if _is_native_case(case):
        versions = _run_pip_resolution(case, tmp_path)
    else:
        versions = _run_uv_resolution(case, tmp_path)
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
