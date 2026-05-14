from typing import Any

from ...blender_operations import ProcessParameter
from ...controlnet_inputs import get_active_controlnet_inputs, get_sdxl_control_modes
from ...model_support import require_supported_sd_version

torch = None
ControlNetModel = None
ControlNetUnionModel = None


def _load_controlnet_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from diffusers import (
            ControlNetModel,
            ControlNetUnionModel,
        )
    except Exception as exc:
        msg = (
            "Python dependencies are not ready in this Blender session. If you just "
            "installed or changed them, restart Blender and try again. Otherwise open "
            "Preferences > Add-ons > DiffusedTexture > Install Python Dependencies, "
            "then restart Blender."
        )
        raise RuntimeError(msg) from exc

    return torch, ControlNetModel, ControlNetUnionModel
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
        return {
            "inputs": inputs,
            "conditioning_scale": process_parameter.union_controlnet_strength,
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
