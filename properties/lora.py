import bpy
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty


class LoRAModel(bpy.types.PropertyGroup):
    path: StringProperty(
        name="LoRA Path", description="Path to the LoRA model file", subtype="FILE_PATH"
    )
    strength: FloatProperty(
        name="LoRA Strength",
        description="Strength of the LoRA model",
        default=1.0,
        min=0.0,
        max=2.0,
    )


def update_loras(self, context):
    scene = context.scene
    num_loras = scene.num_loras
    lora_models = scene.lora_models

    while len(lora_models) < num_loras:
        lora_models.add()
    while len(lora_models) > num_loras:
        lora_models.remove(len(lora_models) - 1)


def register_lora_properties():
    bpy.utils.register_class(LoRAModel)
    bpy.types.Scene.num_loras = IntProperty(
        name="Number of LoRAs",
        description="Number of additional LoRA models to use",
        default=0,
        min=0,
        update=update_loras,
    )
    bpy.types.Scene.lora_models = CollectionProperty(type=LoRAModel)


def unregister_lora_properties():
    del bpy.types.Scene.num_loras
    del bpy.types.Scene.lora_models
    bpy.utils.unregister_class(LoRAModel)
