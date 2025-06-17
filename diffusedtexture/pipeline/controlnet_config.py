from typing import Any

import bpy
import torch
from diffusers import ControlNetModel, ControlNetUnionModel


def build_controlnet_config(context: bpy.types.Context) -> dict[str, dict[str, Any]]:
    """Return a controlnet configuration dictionary for a given scene."""
    if context.scene.sd_version == "sdxl":
        controlnet_model = ControlNetUnionModel.from_pretrained(
            context.scene.controlnet_union_path,
            torch_dtype=torch.float16,
        )
        scale = context.scene.union_controlnet_strength
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
            "conditioning_scale": [context.scene.depth_controlnet_strength],
            "controlnets": [load_model(context.scene.depth_controlnet_path)],
            "inputs": ["depth"],
        },
        "MEDIUM": {
            "conditioning_scale": [
                context.scene.depth_controlnet_strength,
                context.scene.canny_controlnet_strength,
            ],
            "controlnets": [
                load_model(context.scene.depth_controlnet_path),
                load_model(context.scene.canny_controlnet_path),
            ],
            "inputs": ["depth", "canny"],
        },
        "HIGH": {
            "conditioning_scale": [
                context.scene.depth_controlnet_strength,
                context.scene.canny_controlnet_strength,
                context.scene.normal_controlnet_strength,
            ],
            "controlnets": [
                load_model(context.scene.depth_controlnet_path),
                load_model(context.scene.canny_controlnet_path),
                load_model(context.scene.normal_controlnet_path),
            ],
            "inputs": ["depth", "canny", "normal"],
        },
    }
