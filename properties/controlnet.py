import bpy
from bpy.props import (
    FloatProperty,
    StringProperty,
)

from ..model_support import MODEL_SUPPORT_MATRIX


def register_controlnet_properties() -> None:
    """Register ControlNet Properties."""
    bpy.types.Scene.controlnet_union_path = StringProperty(
        name="ControlNet Union Path",
        description="Path to ControlNet Union model (for SDXL)",
        subtype="FILE_PATH",
        default=MODEL_SUPPORT_MATRIX["sdxl"]["controlnet_union_path"],
    )

    bpy.types.Scene.union_controlnet_strength = FloatProperty(
        name="Union ControlNet Strength",
        description="Overall strength for ControlNet Union conditioning",
        default=1.0,
        min=0.0,
        max=1.0,
    )

    bpy.types.Scene.depth_controlnet_path = StringProperty(
        name="Depth ControlNet Path",
        description="Path to depth ControlNet",
        subtype="FILE_PATH",
        default=MODEL_SUPPORT_MATRIX["sd15"]["depth_controlnet_path"],
    )
    bpy.types.Scene.canny_controlnet_path = StringProperty(
        name="Canny ControlNet Path",
        description="Path to canny ControlNet",
        subtype="FILE_PATH",
        default=MODEL_SUPPORT_MATRIX["sd15"]["canny_controlnet_path"],
    )
    bpy.types.Scene.normal_controlnet_path = StringProperty(
        name="Normal ControlNet Path",
        description="Path to normal ControlNet",
        subtype="FILE_PATH",
        default=MODEL_SUPPORT_MATRIX["sd15"]["normal_controlnet_path"],
    )

    bpy.types.Scene.depth_controlnet_strength = FloatProperty(
        name="Depth Strength",
        description="Strength for depth ControlNet",
        default=1.0,
        min=0.0,
        max=1.0,
    )
    bpy.types.Scene.canny_controlnet_strength = FloatProperty(
        name="Canny Strength",
        description="Strength for canny ControlNet",
        default=0.9,
        min=0.0,
        max=1.0,
    )
    bpy.types.Scene.normal_controlnet_strength = FloatProperty(
        name="Normal Strength",
        description="Strength for normal ControlNet",
        default=0.9,
        min=0.0,
        max=1.0,
    )


def unregister_controlnet_properties() -> None:
    for prop_name in (
        "controlnet_union_path",
        "union_controlnet_strength",
        "depth_controlnet_path",
        "canny_controlnet_path",
        "normal_controlnet_path",
        "depth_controlnet_strength",
        "canny_controlnet_strength",
        "normal_controlnet_strength",
    ):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
