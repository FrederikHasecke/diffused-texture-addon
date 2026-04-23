import bpy
from bpy.props import (
    EnumProperty,
    IntProperty,
    StringProperty,
)

from ..model_support import (
    DEFAULT_SD_VERSION,
    MODEL_SUPPORT_MATRIX,
    get_default_model_paths,
    get_sd_version_enum_items,
    require_supported_sd_version,
)


def update_sd_paths(self: bpy.types.Scene, context: bpy.types.Context) -> None:  # noqa: ARG001
    model_version = require_supported_sd_version(context.scene.sd_version)
    defaults = get_default_model_paths(model_version)
    context.scene.checkpoint_path = defaults["checkpoint_path"]
    context.scene.controlnet_union_path = defaults["controlnet_union_path"]
    context.scene.canny_controlnet_path = defaults["canny_controlnet_path"]
    context.scene.normal_controlnet_path = defaults["normal_controlnet_path"]
    context.scene.depth_controlnet_path = defaults["depth_controlnet_path"]


def register_stable_diffusion_properties() -> None:
    bpy.types.Scene.sd_version = EnumProperty(
        name="Stable Diffusion Version",
        description="Choose a supported diffusion model",
        items=get_sd_version_enum_items(),
        default=DEFAULT_SD_VERSION,
        update=update_sd_paths,
    )

    bpy.types.Scene.checkpoint_path = StringProperty(
        name="Checkpoint Path",
        description="Path to the base Stable Diffusion model",
        subtype="FILE_PATH",
        default=MODEL_SUPPORT_MATRIX[DEFAULT_SD_VERSION]["checkpoint_path"],
    )

    bpy.types.Scene.dtype = EnumProperty(
        name="Data Type",
        description="Data type for model inference",
        items=[
            (
                "float16",
                "float16",
                "Use float16 precision",
            ),
            (
                "bfloat16",
                "bfloat16",
                "Use bfloat16 precision (currently unused by supported models)",
            ),
        ],
        default=MODEL_SUPPORT_MATRIX[DEFAULT_SD_VERSION]["default_dtype"],
    )

    bpy.types.Scene.custom_sd_resolution = IntProperty(
        name="Custom SD Resolution",
        description="Optional custom resolution for SD input",
        default=0,
        min=0,
    )


def unregister_stable_diffusion_properties() -> None:
    for prop_name in (
        "sd_version",
        "checkpoint_path",
        "dtype",
        "custom_sd_resolution",
    ):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
