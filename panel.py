import bpy

from .user_interface.main_panel import OBJECT_PT_MainPanel
from .user_interface.lora_panel import OBJECT_PT_LoRAPanel
from .user_interface.ipadapter_panel import OBJECT_PT_IPAdapterPanel
from .user_interface.advanced_panel import OBJECT_PT_AdvancedPanel
from .user_interface.operators import (
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_OT_OpenNewInputImage,
)

classes = [
    OBJECT_PT_MainPanel,
    OBJECT_PT_LoRAPanel,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_PT_AdvancedPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_OT_OpenNewInputImage,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
