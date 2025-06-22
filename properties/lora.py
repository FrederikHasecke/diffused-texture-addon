import bpy  # noqa: I001
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty  # type: ignore  # noqa: PGH003


class LoRAModel(bpy.types.PropertyGroup):
    """LoRA PropertyGroup.

    Args:
        bpy (_type_): _description_
    """

    path: StringProperty(
        name="LoRA Path",
        description="Path to the LoRA model file",
        subtype="FILE_PATH",
    )  # type: ignore  # noqa: PGH003
    strength: FloatProperty(
        name="LoRA Strength",
        description="Strength of the LoRA model",
        default=1.0,
        min=0.0,
        max=2.0,
    )  # type: ignore  # noqa: PGH003


def update_loras(self: bpy.types.Scene, context: bpy.types.Context) -> None:
    num_loras = context.scene.num_loras
    lora_models = context.scene.lora_models

    while len(lora_models) < num_loras:
        lora_models.add()
    while len(lora_models) > num_loras:
        lora_models.remove(len(lora_models) - 1)


def register_lora_properties() -> None:
    """Register all LoRA Panel Properties."""
    bpy.utils.register_class(LoRAModel)
    bpy.types.Scene.num_loras = IntProperty(
        name="Number of LoRAs",
        description="Number of additional LoRA models to use",
        default=0,
        min=0,
        update=update_loras,
    )
    bpy.types.Scene.lora_models = CollectionProperty(type=LoRAModel)


def unregister_lora_properties() -> None:
    del bpy.types.Scene.num_loras
    del bpy.types.Scene.lora_models
    bpy.utils.unregister_class(LoRAModel)
