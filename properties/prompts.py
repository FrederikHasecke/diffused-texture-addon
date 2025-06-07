import bpy
from bpy.props import StringProperty, FloatProperty


def register_prompt_properties():
    bpy.types.Scene.my_prompt = StringProperty(
        name="Prompt", description="Define what the object should be"
    )
    bpy.types.Scene.my_negative_prompt = StringProperty(
        name="Negative Prompt", description="Define what the object should NOT be"
    )
    bpy.types.Scene.guidance_scale = FloatProperty(
        name="Guidance Scale",
        description="Controls alignment with prompt vs. creativity.",
        default=7.0,
        min=0.0,
        max=30.0,
    )


def unregister_prompt_properties():
    del bpy.types.Scene.my_prompt
    del bpy.types.Scene.my_negative_prompt
    del bpy.types.Scene.guidance_scale
