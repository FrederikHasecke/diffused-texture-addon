from typing import Any

import torch
from diffusers import ControlNetModel, ControlNetUnionModel

from ...blender_operations import ProcessParameters


def build_controlnet_config(
    process_parameters: ProcessParameters,
) -> dict[str, dict[str, Any]]:
    """Return a controlnet configuration dictionary for a given scene."""
    if process_parameters.sd_version == "sdxl":
        controlnet_model = ControlNetUnionModel.from_pretrained(
            process_parameters.controlnet_union_path,
            torch_dtype=torch.float16,
        )
        scale = process_parameters.union_controlnet_strength
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

    def load_model(path: str) -> ControlNetModel:
        """Load sd15 cn model."""
        return ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)

    return {
        "LOW": {
            "conditioning_scale": [process_parameters.depth_controlnet_strength],
            "controlnets": [load_model(process_parameters.depth_controlnet_path)],
            "inputs": ["depth"],
        },
        "MEDIUM": {
            "conditioning_scale": [
                process_parameters.depth_controlnet_strength,
                process_parameters.canny_controlnet_strength,
            ],
            "controlnets": [
                load_model(process_parameters.depth_controlnet_path),
                load_model(process_parameters.canny_controlnet_path),
            ],
            "inputs": ["depth", "canny"],
        },
        "HIGH": {
            "conditioning_scale": [
                process_parameters.depth_controlnet_strength,
                process_parameters.canny_controlnet_strength,
                process_parameters.normal_controlnet_strength,
            ],
            "controlnets": [
                load_model(process_parameters.depth_controlnet_path),
                load_model(process_parameters.canny_controlnet_path),
                load_model(process_parameters.normal_controlnet_path),
            ],
            "inputs": ["depth", "canny", "normal"],
        },
    }
