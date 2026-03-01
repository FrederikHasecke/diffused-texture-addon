"""Handles registration and unregistration of all classes and properties for the addon.

This includes UI panels, operators, preferences,
and custom properties in the correct order.
"""

import bpy

from .operators import OBJECT_OT_GenerateTexture
from .preferences import (
    DiffuseTexPreferences,
    InstallDepsOperator,
    InstallModelsOperator,
)
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
    InstallModelsOperator,
    InstallDepsOperator,
    DiffuseTexPreferences,
    OBJECT_OT_GenerateTexture,
    OBJECT_OT_SelectPipette,
    OBJECT_PT_DiffusedTextureMainPanel,
    OBJECT_OT_OpenNewInputImage,
    OBJECT_PT_AdvancedPanel,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_PT_LoRAPanel,
]


def _rollback_registered_classes(registered_classes: list[type]) -> list[str]:
    rollback_errors: list[str] = []
    for cls in reversed(registered_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001
            rollback_errors.append(f"{cls.__name__}: {exc!s}")
    return rollback_errors


def register_addon() -> None:
    """Register all classes and properties in the correct order."""
    registered_classes: list[type] = []

    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as exc:
            rollback_errors = _rollback_registered_classes(registered_classes)
            msg = f"Failed to register class '{cls.__name__}'."
            if rollback_errors:
                rollback_details = "; ".join(rollback_errors)
                msg = f"{msg} Rollback errors: {rollback_details}"
            raise RuntimeError(msg) from exc
        registered_classes.append(cls)

    try:
        register_properties()
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            unregister_properties()
        except Exception as cleanup_exc:  # noqa: BLE001
            rollback_errors.append(f"properties: {cleanup_exc!s}")

        rollback_errors.extend(_rollback_registered_classes(registered_classes))

        msg = "Failed to register addon properties."
        if rollback_errors:
            rollback_details = "; ".join(rollback_errors)
            msg = f"{msg} Rollback errors: {rollback_details}"
        raise RuntimeError(msg) from exc


def unregister_addon() -> None:
    """Unregister all classes and properties in reverse registration order."""
    try:
        unregister_properties()
    except Exception as exc:
        msg = "Failed to unregister addon properties."
        raise RuntimeError(msg) from exc

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:
            msg = f"Failed to unregister class '{cls.__name__}'."
            raise RuntimeError(msg) from exc
