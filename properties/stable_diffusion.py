from enum import Enum

import bpy
from bpy.props import (  # type: ignore  # noqa: PGH003
    EnumProperty,
    IntProperty,
    StringProperty,
)


class stable_diffusion_paths(Enum):
    """Default Setting for SD paths."""

    sd15_ckpt = "runwayml/stable-diffusion-v1-5"
    sd15_cn_canny = "lllyasviel/sd-controlnet-canny"
    sd15_cn_normal = "lllyasviel/sd-controlnet-normal"
    sd15_cn_depth = "lllyasviel/sd-controlnet-depth"
    sdxl_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_cn_union = "xinsir/controlnet-union-sdxl-1.0"


def update_sd_paths(self: bpy.types.Scene, context: bpy.types.Context) -> None:
    if context.scene.sd_version == "sd15":
        context.scene.checkpoint_path = stable_diffusion_paths.sd15_ckpt.value
        context.scene.canny_controlnet_path = stable_diffusion_paths.sd15_cn_canny.value
        context.scene.normal_controlnet_path = (
            stable_diffusion_paths.sd15_cn_normal.value
        )
        context.scene.depth_controlnet_path = stable_diffusion_paths.sd15_cn_depth.value
    elif context.scene.sd_version == "sdxl":
        context.scene.checkpoint_path = stable_diffusion_paths.sdxl_ckpt.value
        context.scene.controlnet_union_path = stable_diffusion_paths.sdxl_cn_union.value
    else:
        msg = (
            "Invalid Stable Diffusion version selected. Please choose 'sd15' or 'sdxl'."
        )
        context.scene.checkpoint_path = ""
        raise ValueError(msg)


def register_stable_diffusion_properties() -> None:
    bpy.types.Scene.sd_version = EnumProperty(
        name="Stable Diffusion Version",
        description="Choose between SD 1.5 or SDXL",
        items=[
            ("sd15", "Stable Diffusion 1.5", ""),
            ("sdxl", "Stable Diffusion XL", ""),
        ],
        default="sd15",
        update=update_sd_paths,
    )

    bpy.types.Scene.checkpoint_path = StringProperty(
        name="Checkpoint Path",
        description="Path to the base Stable Diffusion model",
        subtype="FILE_PATH",
        default="runwayml/stable-diffusion-v1-5",
    )

    bpy.types.Scene.custom_sd_resolution = IntProperty(
        name="Custom SD Resolution",
        description="Optional custom resolution for SD input",
        default=0,
        min=0,
    )


def unregister_stable_diffusion_properties() -> None:
    del bpy.types.Scene.sd_version
    del bpy.types.Scene.checkpoint_path
    del bpy.types.Scene.custom_sd_resolution
