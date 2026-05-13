import pytest

from controlnet_inputs import (
    get_active_controlnet_inputs,
    get_sdxl_control_modes,
    is_image_operation_mode,
)


@pytest.mark.parametrize(
    ("operation_mode", "mesh_complexity", "expected_inputs"),
    [
        ("PARALLEL_IMG", "LOW", ["depth"]),
        ("PARALLEL_IMG", "MEDIUM", ["depth", "canny"]),
        ("PARALLEL_IMG", "HIGH", ["depth", "canny", "normal"]),
        ("SEQUENTIAL_IMG", "LOW", ["depth"]),
        ("PARA_SEQUENTIAL_IMG", "HIGH", ["depth", "canny", "normal"]),
    ],
)
def test_image_modes_keep_existing_controlnet_inputs(
    operation_mode: str,
    mesh_complexity: str,
    expected_inputs: list[str],
) -> None:
    assert get_active_controlnet_inputs(operation_mode, mesh_complexity) == expected_inputs


@pytest.mark.parametrize(
    ("mesh_complexity", "expected_inputs", "expected_sdxl_modes"),
    [
        ("LOW", ["depth"], [1]),
        ("MEDIUM", ["depth", "canny"], [1, 3]),
        ("HIGH", ["depth", "canny", "normal"], [1, 3, 4]),
    ],
)
def test_uv_mode_uses_mode_specific_controlnet_inputs(
    mesh_complexity: str,
    expected_inputs: list[str],
    expected_sdxl_modes: list[int],
) -> None:
    assert get_active_controlnet_inputs("UV_PASS", mesh_complexity) == expected_inputs
    assert get_sdxl_control_modes("UV_PASS", mesh_complexity) == expected_sdxl_modes


def test_uv_mode_is_not_treated_as_image_mode() -> None:
    assert is_image_operation_mode("PARALLEL_IMG") is True
    assert is_image_operation_mode("UV_PASS") is False
