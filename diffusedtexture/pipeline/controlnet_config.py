import torch
from diffusers import ControlNetModel, ControlNetUnionModel


def build_controlnet_config(scene):
    """Return a controlnet configuration dictionary for a given scene."""
    is_sdxl = scene.sd_version == "sdxl"
    use_union = is_sdxl and scene.controlnet_type == "UNION"

    if use_union:
        controlnet_model = ControlNetUnionModel.from_pretrained(
            scene.controlnet_union_path, torch_dtype=torch.float16
        )
        scale = scene.union_controlnet_strength
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

    def load_model(path):
        return ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)

    return {
        "LOW": {
            "conditioning_scale": [scene.depth_controlnet_strength],
            "controlnets": [load_model(scene.depth_controlnet_path)],
            "inputs": ["depth"],
        },
        "MEDIUM": {
            "conditioning_scale": [
                scene.depth_controlnet_strength,
                scene.canny_controlnet_strength,
            ],
            "controlnets": [
                load_model(scene.depth_controlnet_path),
                load_model(scene.canny_controlnet_path),
            ],
            "inputs": ["depth", "canny"],
        },
        "HIGH": {
            "conditioning_scale": [
                scene.depth_controlnet_strength,
                scene.canny_controlnet_strength,
                scene.normal_controlnet_strength,
            ],
            "controlnets": [
                load_model(scene.depth_controlnet_path),
                load_model(scene.canny_controlnet_path),
                load_model(scene.normal_controlnet_path),
            ],
            "inputs": ["depth", "canny", "normal"],
        },
    }
