"""Handles registration and unregistration of all classes and properties for the addon.

This includes UI panels, operators, preferences,
and custom properties in the correct order.
"""

import bpy

from .operators import OBJECT_OT_GenerateTexture
from .preferences import DiffuseTexPreferences, InstallModelsOperator
from .properties import register_properties, unregister_properties
from .user_interface.advanced_panel import OBJECT_PT_AdvancedPanel
from .user_interface.ipadapter_panel import (
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_PT_IPAdapterPanel,
)
from .user_interface.lora_panel import OBJECT_PT_LoRAPanel
from .user_interface.main_panel import (
    OBJECT_OT_OpenNewInputImage,
    OBJECT_OT_SelectPipette,
    OBJECT_PT_DiffusedTextureMainPanel,
)

classes = [
    DiffuseTexPreferences,
    InstallModelsOperator,
    OBJECT_OT_GenerateTexture,
    OBJECT_OT_SelectPipette,
    OBJECT_PT_DiffusedTextureMainPanel,
    OBJECT_OT_OpenNewInputImage,
    OBJECT_PT_AdvancedPanel,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_PT_LoRAPanel,
]


def register_addon() -> None:
    """Register all  and properties in the correct order."""
    for cls in classes:
        bpy.utils.register_class(cls)
    register_properties()


def unregister_addon() -> None:
    """Unegister all  and properties in the correct order."""
    unregister_properties()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
