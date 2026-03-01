from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from diffusers import (
        ControlNetModel,
        ControlNetUnionModel,
        QwenImageMultiControlNetModel,
    )
except ModuleNotFoundError:
    ControlNetModel = None
    ControlNetUnionModel = None
    QwenImageMultiControlNetModel = None

from ...blender_operations import ProcessParameter


def build_controlnet_config(
    process_parameter: ProcessParameter,
) -> dict[str, dict[str, Any]]:
    """Return a controlnet configuration dictionary for a given scene."""
    if process_parameter.sd_version == "sdxl":
        torch_dtype = torch.float16
        if process_parameter.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        controlnet_model = ControlNetUnionModel.from_pretrained(
            process_parameter.controlnet_union_path,
            torch_dtype=torch_dtype,
        )
        scale = process_parameter.union_controlnet_strength
        return {
            "LOW": {
                "union_control": True,
                "control_mode": [1],
                "controlnets": controlnet_model,
                "conditioning_scale": scale,
                "inputs": ["depth"],
            },
            "MEDIUM": {
                "union_control": True,
                "control_mode": [1, 3],
                "controlnets": controlnet_model,
                "conditioning_scale": scale,
                "inputs": ["depth", "canny"],
            },
            "HIGH": {
                "union_control": True,
                "control_mode": [1, 3, 4],
                "controlnets": controlnet_model,
                "conditioning_scale": scale,
                "inputs": ["depth", "canny", "normal"],
            },
        }

    if process_parameter.sd_version == "sd15":
        torch_dtype = torch.float16
        if process_parameter.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        def load_model(path: str) -> ControlNetModel | None:  # type: ignore  # noqa: PGH003
            """Load sd15 cn model."""
            return ControlNetModel.from_pretrained(path, torch_dtype=torch_dtype)

        return {
            "LOW": {
                "conditioning_scale": [process_parameter.depth_controlnet_strength],
                "controlnets": [load_model(process_parameter.depth_controlnet_path)],
                "inputs": ["depth"],
            },
            "MEDIUM": {
                "conditioning_scale": [
                    process_parameter.depth_controlnet_strength,
                    process_parameter.canny_controlnet_strength,
                ],
                "controlnets": [
                    load_model(process_parameter.depth_controlnet_path),
                    load_model(process_parameter.canny_controlnet_path),
                ],
                "inputs": ["depth", "canny"],
            },
            "HIGH": {
                "conditioning_scale": [
                    process_parameter.depth_controlnet_strength,
                    process_parameter.canny_controlnet_strength,
                    process_parameter.normal_controlnet_strength,
                ],
                "controlnets": [
                    load_model(process_parameter.depth_controlnet_path),
                    load_model(process_parameter.canny_controlnet_path),
                    load_model(process_parameter.normal_controlnet_path),
                ],
                "inputs": ["depth", "canny", "normal"],
            },
        }

    if process_parameter.sd_version == "flux":
        # TODO: Implement flux pipeline
        msg = "Flux model is not yet implemented."
        raise NotImplementedError(msg)

    if process_parameter.sd_version == "qwen":
        torch_dtype = torch.bfloat16
        if process_parameter.dtype == "float16":
            torch_dtype = torch.float16

        controlnet_model = QwenImageMultiControlNetModel.from_pretrained(
            process_parameter.controlnet_union_path,
            torch_dtype=torch_dtype,
        )
        scale = process_parameter.union_controlnet_strength
        return {
            "LOW": {
                "controlnet_conditioning_scale": [
                    process_parameter.depth_controlnet_strength,
                ],
                "controlnet": controlnet_model,
                "inputs": ["depth"],
            },
            "MEDIUM": {
                "controlnet_conditioning_scale": [
                    process_parameter.depth_controlnet_strength,
                    process_parameter.canny_controlnet_strength,
                ],
                "controlnet": controlnet_model,
                "inputs": ["depth", "canny"],
            },
            "HIGH": {
                "controlnet_conditioning_scale": [
                    process_parameter.depth_controlnet_strength,
                    process_parameter.canny_controlnet_strength,
                    process_parameter.normal_controlnet_strength,
                ],
                "controlnet": controlnet_model,
                "inputs": ["depth", "canny", "normal"],
            },
        }

    msg = "Invalid Stable Diffusion version selected."
    raise ValueError(
        msg,
    )
