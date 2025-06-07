import bpy
from bpy.props import StringProperty, EnumProperty, IntProperty


def update_sd_paths(self, context):
    if context.scene.sd_version == "sd15":
        context.scene.checkpoint_path = "runwayml/stable-diffusion-v1-5"
        context.scene.canny_controlnet_path = "lllyasviel/sd-controlnet-canny"
        context.scene.normal_controlnet_path = "lllyasviel/sd-controlnet-normal"
        context.scene.depth_controlnet_path = "lllyasviel/sd-controlnet-depth"
    elif context.scene.sd_version == "sdxl":
        context.scene.checkpoint_path = "stabilityai/stable-diffusion-xl-base-1.0"
        context.scene.canny_controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
        context.scene.normal_controlnet_path = ""
        context.scene.depth_controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
        context.scene.controlnet_union_path = "xinsir/controlnet-union-sdxl-1.0"


def register_stable_diffusion_properties():
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


def unregister_stable_diffusion_properties():
    del bpy.types.Scene.sd_version
    del bpy.types.Scene.checkpoint_path
    del bpy.types.Scene.custom_sd_resolution
