import bpy
from bpy.props import FloatProperty, StringProperty


def register_prompt_properties() -> None:
    bpy.types.Scene.my_prompt = StringProperty(
        name="Prompt",
        description="Define what the object should be",
    )
    bpy.types.Scene.my_negative_prompt = StringProperty(
        name="Negative Prompt",
        description="Define what the object should NOT be",
    )
    bpy.types.Scene.guidance_scale = FloatProperty(
        name="Guidance Scale",
        description="Controls alignment with prompt vs. creativity.",
        default=7.0,
        min=0.0,
        max=30.0,
    )


def unregister_prompt_properties() -> None:
    for prop_name in ("my_prompt", "my_negative_prompt", "guidance_scale"):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
