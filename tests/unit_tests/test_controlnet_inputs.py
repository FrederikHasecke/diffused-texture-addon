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
    "operation_mode",
    ["UV_PASS", "UNKNOWN_MODE"],
)
def test_unsupported_modes_raise_clear_errors(operation_mode: str) -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_active_controlnet_inputs(operation_mode, "HIGH")

    with pytest.raises(ValueError, match="not supported"):
        get_sdxl_control_modes(operation_mode, "HIGH")


@pytest.mark.parametrize(
    ("mesh_complexity", "expected_sdxl_modes"),
    [
        ("LOW", [1]),
        ("MEDIUM", [1, 3]),
        ("HIGH", [1, 3, 4]),
    ],
)
def test_image_modes_keep_existing_sdxl_control_modes(
    mesh_complexity: str,
    expected_sdxl_modes: list[int],
) -> None:
    assert get_sdxl_control_modes("PARALLEL_IMG", mesh_complexity) == expected_sdxl_modes


def test_only_supported_image_modes_are_treated_as_image_modes() -> None:
    assert is_image_operation_mode("PARALLEL_IMG") is True
    assert is_image_operation_mode("SEQUENTIAL_IMG") is True
    assert is_image_operation_mode("PARA_SEQUENTIAL_IMG") is True
    assert is_image_operation_mode("UV_PASS") is False
