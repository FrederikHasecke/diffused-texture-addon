import os
import bpy
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    CollectionProperty,
    EnumProperty,
    BoolProperty,
)
from utils import update_uv_maps, get_mesh_objects


class LoRAModel(bpy.types.PropertyGroup):
    path: StringProperty(
        name="LoRA Path", description="Path to the LoRA model file", subtype="FILE_PATH"
    )
    strength: FloatProperty(
        name="Strength LoRA",
        description="Strength of the LoRA model",
        default=1.0,
        min=0.0,
        max=2.0,
    )


def get_image_list(self, context):
    images = [("", "None", "No image selected")]  # Default option with no selection
    images += [(img.name, img.name, "") for img in bpy.data.images]
    return images


def update_loras(self, context):
    scene = context.scene
    num_loras = scene.num_loras
    lora_models = scene.lora_models

    while len(lora_models) < num_loras:
        lora_models.add()

    while len(lora_models) > num_loras:
        lora_models.remove(len(lora_models) - 1)


def register_properties():
    bpy.utils.register_class(LoRAModel)

    bpy.types.Scene.my_mesh_object = EnumProperty(
        name="Mesh Object",
        items=get_mesh_objects,
        description="Select the mesh object you want to use texturize.",
    )

    bpy.types.Scene.my_uv_map = EnumProperty(
        name="UV Map",
        items=update_uv_maps,
        description="Select the UV map you want to use for the final texture.",
    )

    bpy.types.Scene.my_prompt = StringProperty(
        name="Prompt", description="Define what the object should be"
    )

    bpy.types.Scene.my_negative_prompt = StringProperty(
        name="Negative Prompt", description="Define what the object should NOT be"
    )

    bpy.types.Scene.mesh_complexity = EnumProperty(
        name="Mesh Complexity",
        description="The complexity, polycount and detail of the selected mesh.",
        items=[("LOW", "Low", ""), ("MEDIUM", "Medium", ""), ("HIGH", "High", "")],
    )

    bpy.types.Scene.texture_resolution = EnumProperty(
        name="Texture Resolution",
        description="The final texture resolution of the selected mesh object.",
        items=[
            ("512", "512x512", ""),
            ("1024", "1024x1024", ""),
            ("2048", "2048x2048", ""),
            ("4096", "4096x4096", ""),
        ],
    )

    bpy.types.Scene.output_path = StringProperty(
        name="Output Path",
        description="Directory to store the resulting texture and temporary files",
        subtype="DIR_PATH",
        default=os.path.dirname(bpy.data.filepath) if bpy.data.filepath else "",
    )

    bpy.types.Scene.input_texture_path = StringProperty(
        name="Input Texture",
        description="Select an input texture file for img2img or texture2texture pass.",
        subtype="FILE_PATH",
    )

    bpy.types.Scene.first_pass = BoolProperty(
        name="First Pass from Gaussian Noise",
        description="First pass creates one result from multiple viewpoints at once.",
    )

    bpy.types.Scene.second_pass = BoolProperty(
        name="Second Pass in Image Space",
        description="Second pass improves results from multiple viewpoints successively.",
    )

    bpy.types.Scene.refinement_uv_space = BoolProperty(
        name="Refinement in UV Space",
        description="Refinement in UV space fixes areas not visible to the camera or holes in the texture.",
    )

    bpy.types.Scene.texture_seed = IntProperty(
        name="Seed",
        description="Seed for randomization to ensure repeatable results",
        default=0,
        min=0,
    )

    bpy.types.Scene.checkpoint_path = StringProperty(
        name="Checkpoint Path",
        description="Optional path to the Stable Diffusion base model checkpoint",
        subtype="FILE_PATH",
    )

    bpy.types.Scene.num_loras = IntProperty(
        name="Number of LoRAs",
        description="Number of additional LoRA models to use",
        default=0,
        min=0,
        update=update_loras,
    )
    bpy.types.Scene.lora_models = CollectionProperty(type=LoRAModel)

    # IPAdapter-specific properties
    bpy.types.Scene.use_ipadapter = BoolProperty(
        name="Use IPAdapter",
        description="Activate and use IPAdapter during the texture generation process.",
        default=False,
    )

    bpy.types.Scene.ipadapter_image_path = StringProperty(
        name="Image Path IPAdapter",
        description="Path to the image for the IPAdapter process",
        subtype="FILE_PATH",
    )

    bpy.types.Scene.ipadapter_model_path = StringProperty(
        name="Model Path IPAdapter",
        description="Optional path to the IPAdapter pretrained model",
        subtype="FILE_PATH",
    )

    bpy.types.Scene.ipadapter_strength = FloatProperty(
        name="IPAdapter Strength",
        description="Strength of the IPAdapter effect",
        default=1.0,
        min=0.0,
        soft_max=10.0,
    )


def unregister_properties():
    bpy.utils.unregister_class(LoRAModel)

    del bpy.types.Scene.num_loras
    del bpy.types.Scene.lora_models
    del bpy.types.Scene.ipadapter_image_path
    del bpy.types.Scene.ipadapter_model_path
    del bpy.types.Scene.ipadapter_strength

    del bpy.types.Scene.my_mesh_object
    del bpy.types.Scene.my_uv_map
    del bpy.types.Scene.selected_image
    del bpy.types.Scene.my_prompt
    del bpy.types.Scene.my_negative_prompt
    del bpy.types.Scene.mesh_complexity
    del bpy.types.Scene.texture_resolution
    del bpy.types.Scene.output_path
    del bpy.types.Scene.first_pass
    del bpy.types.Scene.second_pass
    del bpy.types.Scene.refinement_uv_space
    del bpy.types.Scene.texture_seed
    del bpy.types.Scene.checkpoint_path
