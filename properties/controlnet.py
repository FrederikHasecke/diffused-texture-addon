import bpy
from bpy.props import EnumProperty, StringProperty, FloatProperty


def register_controlnet_properties():
    bpy.types.Scene.controlnet_type = EnumProperty(
        name="ControlNet Type",
        description="Choose between traditional or union-style ControlNet",
        items=[
            (
                "MULTIPLE",
                "Multiple ControlNets",
                "Use multiple separate ControlNet models",
            ),
            ("UNION", "ControlNet Union", "Use a single ControlNet Union model"),
        ],
        default="MULTIPLE",
    )

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
        default="lllyasviel/sd-controlnet-depth",
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


def unregister_controlnet_properties():
    del bpy.types.Scene.controlnet_type
    del bpy.types.Scene.controlnet_union_path
    del bpy.types.Scene.union_controlnet_strength
    del bpy.types.Scene.depth_controlnet_path
    del bpy.types.Scene.canny_controlnet_path
    del bpy.types.Scene.normal_controlnet_path
    del bpy.types.Scene.depth_controlnet_strength
    del bpy.types.Scene.canny_controlnet_strength
    del bpy.types.Scene.normal_controlnet_strength
