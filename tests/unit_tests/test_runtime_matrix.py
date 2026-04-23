from pathlib import Path

import pytest

from installer.runtime_matrix import (
    constraints_filename,
    expected_python_version,
    normalize_python_version,
    resolve_runtime_spec,
    resolve_torch_install,
    torch_index_url,
)


def test_expected_python_version_switches_at_blender_51() -> None:
    assert expected_python_version((4, 2, 0)) == (3, 11)
    assert expected_python_version((5, 0, 9)) == (3, 11)
    assert expected_python_version((5, 1, 0)) == (3, 13)


def test_constraints_filename_matches_supported_python_series() -> None:
    assert constraints_filename((3, 11)) == "runtime-py311.txt"
    assert constraints_filename((3, 13)) == "runtime-py313.txt"


def test_normalize_python_version_rejects_unsupported_versions() -> None:
    with pytest.raises(ValueError, match="Unsupported Python version"):
        normalize_python_version((3, 12))


def test_runtime_spec_for_blender_50_uses_numpy_1_constraints() -> None:
    spec = resolve_runtime_spec(blender_version=(5, 0, 0), python_version=(3, 11))

    assert spec.python_tag == "py311"
    assert spec.constraints_path.name == "runtime-py311.txt"
    assert "numpy>=1.26,<2.0" in spec.runtime_requirements
    assert "numpy>=2,<3" not in spec.runtime_requirements


def test_runtime_spec_for_blender_51_uses_numpy_2_constraints() -> None:
    spec = resolve_runtime_spec(blender_version=(5, 1, 0), python_version=(3, 13))

    assert spec.python_tag == "py313"
    assert spec.constraints_path.name == "runtime-py313.txt"
    assert "numpy>=2.3,<2.4" in spec.runtime_requirements
    assert "numpy>=1.26,<2.0" not in spec.runtime_requirements


def test_runtime_constraints_files_exist() -> None:
    spec_311 = resolve_runtime_spec(blender_version=(4, 2, 0), python_version=(3, 11))
    spec_313 = resolve_runtime_spec(blender_version=(5, 1, 0), python_version=(3, 13))

    assert spec_311.constraints_path.is_file()
    assert spec_313.constraints_path.is_file()
    assert spec_311.constraints_path.parent == spec_313.constraints_path.parent


def test_resolve_torch_install_pins_windows_blender_50() -> None:
    channel, requirement, note = resolve_torch_install(
        "cu130",
        platform="win32",
        blender_version=(5, 0, 0),
    )

    assert channel == "cu129"
    assert requirement == "torch==2.8.0+cu129"
    assert "Pinned PyTorch to 2.8.0" in note


def test_resolve_torch_install_keeps_latest_windows_blender_51() -> None:
    channel, requirement, note = resolve_torch_install(
        "cu130",
        platform="win32",
        blender_version=(5, 1, 0),
    )

    assert channel == "cu130"
    assert requirement == "torch"
    assert note == ""


def test_torch_index_url_maps_expected_channels() -> None:
    assert torch_index_url("cpu") == (
        "https://download.pytorch.org/whl/cpu",
        "PyTorch CPU",
    )
    assert torch_index_url("cu130") == (
        "https://download.pytorch.org/whl/cu130",
        "PyTorch CU130",
    )
    assert torch_index_url("rocm6.3") == (
        "https://download.pytorch.org/whl/rocm6.3",
        "PyTorch ROCm 6.3",
    )
