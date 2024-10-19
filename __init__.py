import bpy
from .panel import OBJECT_PT_MainPanel
from .operators import OBJECT_OT_GenerateTexture, OBJECT_OT_SelectPipette
from .properties import register_properties, unregister_properties

__all__ = ["first_pass", "second_pass", "uv_pass"]


def register():
    register_properties()
    bpy.utils.register_class(OBJECT_PT_MainPanel)
    bpy.utils.register_class(OBJECT_OT_GenerateTexture)
    bpy.utils.register_class(OBJECT_OT_SelectPipette)


def unregister():
    unregister_properties()
    bpy.utils.unregister_class(OBJECT_PT_MainPanel)
    bpy.utils.unregister_class(OBJECT_OT_GenerateTexture)
    bpy.utils.unregister_class(OBJECT_OT_SelectPipette)


if __name__ == "__main__":
    register()
