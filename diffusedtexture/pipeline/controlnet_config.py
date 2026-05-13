from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from diffusers import (
        ControlNetModel,
        ControlNetUnionModel,
    )
except ModuleNotFoundError:
    ControlNetModel = None
    ControlNetUnionModel = None

from ...blender_operations import ProcessParameter
from ...controlnet_inputs import get_active_controlnet_inputs, get_sdxl_control_modes
from ...model_support import require_supported_sd_version

UV_CONTROLNET_STRENGTH_LIMITS = {
    "depth": 0.5,
    "canny": 0.3,
    "normal": 0.55,
}
UV_UNION_CONTROLNET_STRENGTH_LIMIT = 0.65


def _load_controlnet_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from diffusers import (
            ControlNetModel,
            ControlNetUnionModel,
        )
    except ModuleNotFoundError as exc:
        msg = (
            "Python dependencies are not ready in this Blender session. If you just "
            "installed or changed them, restart Blender and try again. Otherwise open "
            "Preferences > Add-ons > DiffusedTexture > Install Python Dependencies, "
            "then restart Blender."
        )
        raise RuntimeError(msg) from exc

    return torch, ControlNetModel, ControlNetUnionModel


def _limited_uv_strength(value: float | None, limit: float) -> float:
    if value is None:
        return limit
    return min(float(value), limit)


def get_controlnet_static_config(
    process_parameter: ProcessParameter,
) -> dict[str, Any]:
    """Return lightweight config (inputs, conditioning_scale) without loading models."""
    model_version = require_supported_sd_version(process_parameter.sd_version)
    inputs = get_active_controlnet_inputs(
        process_parameter.operation_mode,
        process_parameter.mesh_complexity,
    )

    if model_version == "sdxl":
        conditioning_scale = process_parameter.union_controlnet_strength
        if process_parameter.operation_mode == "UV_PASS":
            conditioning_scale = _limited_uv_strength(
                conditioning_scale,
                UV_UNION_CONTROLNET_STRENGTH_LIMIT,
            )
        return {
            "inputs": inputs,
            "conditioning_scale": conditioning_scale,
            "control_mode": get_sdxl_control_modes(
                process_parameter.operation_mode,
                process_parameter.mesh_complexity,
            ),
        }

    if model_version == "sd15":
        strength_map = {
            "depth": process_parameter.depth_controlnet_strength,
            "canny": process_parameter.canny_controlnet_strength,
            "normal": process_parameter.normal_controlnet_strength,
        }
        if process_parameter.operation_mode == "UV_PASS":
            strength_map = {
                key: _limited_uv_strength(value, UV_CONTROLNET_STRENGTH_LIMITS[key])
                for key, value in strength_map.items()
            }
        return {
            "inputs": inputs,
            "conditioning_scale": [strength_map[key] for key in inputs],
        }

    msg = f"Unexpected supported model: {model_version}"
    raise RuntimeError(msg)


def load_controlnet_models(
    process_parameter: ProcessParameter,
) -> Any:  # noqa: ANN401
    """Load ControlNet models for the selected complexity level only."""
    torch, controlnet_model_cls, controlnet_union_model_cls = _load_controlnet_runtime()

    model_version = require_supported_sd_version(process_parameter.sd_version)
    inputs = get_active_controlnet_inputs(
        process_parameter.operation_mode,
        process_parameter.mesh_complexity,
    )

    torch_dtype = torch.float16
    if process_parameter.dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    if model_version == "sdxl":
        return controlnet_union_model_cls.from_pretrained(
            process_parameter.controlnet_union_path,
            torch_dtype=torch_dtype,
        )

    if model_version == "sd15":
        path_map = {
            "depth": process_parameter.depth_controlnet_path,
            "canny": process_parameter.canny_controlnet_path,
            "normal": process_parameter.normal_controlnet_path,
        }
        return [
            controlnet_model_cls.from_pretrained(path_map[key], torch_dtype=torch_dtype)
            for key in inputs
        ]

    msg = f"Unexpected supported model: {model_version}"
    raise RuntimeError(msg)
