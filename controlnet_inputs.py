from typing import Literal, NoReturn, cast

OperationMode = Literal[
    "PARALLEL_IMG",
    "SEQUENTIAL_IMG",
    "PARA_SEQUENTIAL_IMG",
    "UV_PASS",
]
MeshComplexity = Literal["LOW", "MEDIUM", "HIGH"]
ControlInput = Literal["depth", "canny", "normal"]

IMAGE_MODE_OPERATION_MODES = frozenset(
    {"PARALLEL_IMG", "SEQUENTIAL_IMG", "PARA_SEQUENTIAL_IMG"},
)

_IMAGE_MODE_INPUTS: dict[MeshComplexity, list[ControlInput]] = {
    "LOW": ["depth"],
    "MEDIUM": ["depth", "canny"],
    "HIGH": ["depth", "canny", "normal"],
}

_UV_MODE_INPUTS: dict[MeshComplexity, list[ControlInput]] = {
    "LOW": ["depth"],
    "MEDIUM": ["depth", "canny"],
    "HIGH": ["depth", "canny", "normal"],
}

_IMAGE_MODE_SDXL_CONTROL_MODES: dict[MeshComplexity, list[int]] = {
    "LOW": [1],
    "MEDIUM": [1, 3],
    "HIGH": [1, 3, 4],
}

_UV_MODE_SDXL_CONTROL_MODES: dict[MeshComplexity, list[int]] = {
    "LOW": [1],
    "MEDIUM": [1, 3],
    "HIGH": [1, 3, 4],
}


def is_uv_operation_mode(operation_mode: str) -> bool:
    return operation_mode == "UV_PASS"


def is_image_operation_mode(operation_mode: str) -> bool:
    return operation_mode in IMAGE_MODE_OPERATION_MODES


def _raise_unsupported_controlnet_selection(
    operation_mode: str,
    mesh_complexity: str,
) -> NoReturn:
    msg = (
        "Unsupported ControlNet selection for "
        f"operation_mode={operation_mode!r}, mesh_complexity={mesh_complexity!r}."
    )
    raise ValueError(msg)


def _raise_unsupported_sdxl_control_mode(
    operation_mode: str,
    mesh_complexity: str,
) -> NoReturn:
    msg = (
        "Unsupported SDXL ControlNet mode selection for "
        f"operation_mode={operation_mode!r}, mesh_complexity={mesh_complexity!r}."
    )
    raise ValueError(msg)


def _mesh_complexity_key(mesh_complexity: str) -> MeshComplexity:
    if mesh_complexity in _IMAGE_MODE_INPUTS:
        return cast("MeshComplexity", mesh_complexity)

    _raise_unsupported_controlnet_selection("unknown", mesh_complexity)


def get_active_controlnet_inputs(
    operation_mode: str,
    mesh_complexity: str,
) -> list[ControlInput]:
    complexity_key = _mesh_complexity_key(mesh_complexity)
    if is_uv_operation_mode(operation_mode):
        return list(_UV_MODE_INPUTS[complexity_key])
    if is_image_operation_mode(operation_mode):
        return list(_IMAGE_MODE_INPUTS[complexity_key])
    _raise_unsupported_controlnet_selection(operation_mode, mesh_complexity)


def get_sdxl_control_modes(
    operation_mode: str,
    mesh_complexity: str,
) -> list[int]:
    complexity_key = _mesh_complexity_key(mesh_complexity)
    if is_uv_operation_mode(operation_mode):
        return list(_UV_MODE_SDXL_CONTROL_MODES[complexity_key])
    if is_image_operation_mode(operation_mode):
        return list(_IMAGE_MODE_SDXL_CONTROL_MODES[complexity_key])
    _raise_unsupported_sdxl_control_mode(operation_mode, mesh_complexity)
