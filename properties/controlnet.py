import bpy
from bpy.props import (  # type: ignore  # noqa: PGH003
    EnumProperty,
    FloatProperty,
    StringProperty,
)


def register_controlnet_properties() -> None:
    """Register ControlNet Properties."""
    bpy.types.Scene.controlnet_union_path = StringProperty(
        name="ControlNet Union Path",
        description="Path to ControlNet Union model (for SDXL)",
        subtype="FILE_PATH",
        default="xinsir/controlnet-union-sdxl-1.0",
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
        default="lllyasviel/control_v11f1p_sd15_depth",
    )
    bpy.types.Scene.canny_controlnet_path = StringProperty(
        name="Canny ControlNet Path",
        description="Path to canny ControlNet",
        subtype="FILE_PATH",
        default="lllyasviel/sd-controlnet-canny",
    )
    bpy.types.Scene.normal_controlnet_path = StringProperty(
        name="Normal ControlNet Path",
        description="Path to normal ControlNet",
        subtype="FILE_PATH",
        default="lllyasviel/sd-controlnet-normal",
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
    del bpy.types.Scene.controlnet_union_path
    del bpy.types.Scene.union_controlnet_strength
    del bpy.types.Scene.depth_controlnet_path
    del bpy.types.Scene.canny_controlnet_path
    del bpy.types.Scene.normal_controlnet_path
    del bpy.types.Scene.depth_controlnet_strength
    del bpy.types.Scene.canny_controlnet_strength
    del bpy.types.Scene.normal_controlnet_strength
