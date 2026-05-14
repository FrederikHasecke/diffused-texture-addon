from typing import Literal, NoReturn, cast

if __package__:
    from .operation_mode import is_image_operation_mode as _is_image_operation_mode
    from .operation_mode import validate_operation_mode
else:
    from operation_mode import is_image_operation_mode as _is_image_operation_mode
    from operation_mode import validate_operation_mode

MeshComplexity = Literal["LOW", "MEDIUM", "HIGH"]
ControlInput = Literal["depth", "canny", "normal"]

_IMAGE_MODE_INPUTS: dict[MeshComplexity, list[ControlInput]] = {
    "LOW": ["depth"],
    "MEDIUM": ["depth", "canny"],
    "HIGH": ["depth", "canny", "normal"],
}

_IMAGE_MODE_SDXL_CONTROL_MODES: dict[MeshComplexity, list[int]] = {
    "LOW": [1],
    "MEDIUM": [1, 3],
    "HIGH": [1, 3, 4],
}


def is_image_operation_mode(operation_mode: str) -> bool:
    return _is_image_operation_mode(operation_mode)


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
    validate_operation_mode(operation_mode)
    return list(_IMAGE_MODE_INPUTS[complexity_key])


def get_sdxl_control_modes(
    operation_mode: str,
    mesh_complexity: str,
) -> list[int]:
    complexity_key = _mesh_complexity_key(mesh_complexity)
    validate_operation_mode(operation_mode)
    return list(_IMAGE_MODE_SDXL_CONTROL_MODES[complexity_key])
