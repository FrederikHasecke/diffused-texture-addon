import bpy
from bpy.props import (  # type: ignore  # noqa: PGH003
    EnumProperty,
    IntProperty,
    StringProperty,
)

from ..config.config_parameters import stable_diffusion_paths


def update_sd_paths(context: bpy.types.Context) -> None:
    if context.scene.sd_version == "sd15":
        context.scene.checkpoint_path = stable_diffusion_paths.sd15_ckpt
        context.scene.canny_controlnet_path = stable_diffusion_paths.sd15_cn_canny
        context.scene.normal_controlnet_path = stable_diffusion_paths.sd15_cn_normal
        context.scene.depth_controlnet_path = stable_diffusion_paths.sd15_cn_depth
    elif context.scene.sd_version == "sdxl":
        context.scene.checkpoint_path = stable_diffusion_paths.sdxl_ckpt
        context.scene.controlnet_union_path = stable_diffusion_paths.sdxl_cn_union


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
