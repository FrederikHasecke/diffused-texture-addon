import os
import bpy
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    CollectionProperty,
    EnumProperty,
)
from .utils import update_uv_maps, get_mesh_objects


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


def update_loras(self, context):
    scene = context.scene
    num_loras = scene.num_loras
    lora_models = scene.lora_models

    while len(lora_models) < num_loras:
        lora_models.add()

    while len(lora_models) > num_loras:
        lora_models.remove(len(lora_models) - 1)


def update_operation_mode(self, context):
    """Automatically adjust denoise strength based on the operation mode."""
    if self.operation_mode in ["IMAGE2IMAGE_PARALLEL", "IMAGE2IMAGE_SEQUENTIAL"]:
        self.denoise_strength = 0.4
    else:
        self.denoise_strength = 1.0


def update_ipadapter_image(self, context):
    """Ensure the selected image from the preview window is set in scene.ipadapter_image."""
    image = context.scene.ipadapter_image
    if image:
        image_data = bpy.data.images.get(image.name)

        # Only set the image if it's not already correctly set to prevent recursion
        if image_data != context.scene.ipadapter_image:
            context.scene.ipadapter_image = image_data


def update_output_path(self, context):
    if self.output_path.startswith("//"):
        self.output_path = bpy.path.abspath(self.output_path)


def update_input_texture_path(self, context):
    if self.input_texture_path.startswith("//"):
        self.input_texture_path = bpy.path.abspath(self.input_texture_path)


def register_properties():

    try:
        bpy.utils.register_class(LoRAModel)
    except Exception as e:
        print(
            f"Warning: {LoRAModel.__name__} was not registered or failed to register. {e}"
        )

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

    bpy.types.Scene.guidance_scale = FloatProperty(
        name="Guidance Scale",
        description="A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.",
        default=10.0,
        min=0.0,
        max=30.0,  # Ensure this is a float value for finer control
    )

    bpy.types.Scene.operation_mode = EnumProperty(
        name="Operation Mode",
        description="The complexity, polycount and detail of the selected mesh.",
        items=[
            (
                "TEXT2IMAGE_PARALLEL",
                "Text2Image Parallel",
                "Generate textures using text prompts in parallel.",
            ),
            (
                "IMAGE2IMAGE_PARALLEL",
                "Image2Image Parallel",
                "Generate textures using input images in parallel.",
            ),
            (
                "IMAGE2IMAGE_SEQUENTIAL",
                "Image2Image Sequential",
                "Generate textures using input images sequentially.",
            ),
            # (
            #     "TEXTURE2TEXTURE_ENHANCEMENT",
            #     "Texture2Texture Enhancement",
            #     "Enhance textures using input textures.",
            # ),
        ],
        update=update_operation_mode,  # Add the update function
    )

    bpy.types.Scene.denoise_strength = FloatProperty(
        name="Denoise Strength",
        description="Strength of denoise for Stable Diffusion",
        default=1.0,
        min=0.0,
        max=1.0,  # Ensure this is a float value for finer control
    )

    bpy.types.Scene.texture_resolution = EnumProperty(
        name="Texture Resolution",
        description="The final texture resolution of the selected mesh object.",
        items=[
            ("256", "256x256", ""),
            ("512", "512x512", ""),
            ("1024", "1024x1024", ""),
            ("2048", "2048x2048", ""),
            ("4096", "4096x4096", ""),
        ],
    )

    bpy.types.Scene.render_resolution = EnumProperty(
        name="Render Resolution",
        description="The Render resolution used in Stable Diffusion.",
        items=[
            ("512", "512x512", ""),
            ("1024", "1024x1024", ""),
            ("2048", "2048x2048", ""),
            ("4096", "4096x4096", ""),
            ("8192", "8192x8192", ""),
        ],
    )

    bpy.types.Scene.output_path = StringProperty(
        name="Output Path",
        description="Directory to store the resulting texture and temporary files",
        subtype="DIR_PATH",
        default="",
        update=update_output_path,
    )

    bpy.types.Scene.input_texture_path = StringProperty(
        name="Input Texture",
        description="Select an input texture file for img2img or texture2texture pass.",
        subtype="FILE_PATH",
        default="",
        update=update_input_texture_path,
    )

    bpy.types.Scene.mesh_complexity = EnumProperty(
        name="Mesh Complexity",
        description="How complex is the mesh.",
        items=[("LOW", "Low", ""), ("MEDIUM", "Medium", ""), ("HIGH", "High", "")],
    )

    bpy.types.Scene.num_cameras = EnumProperty(
        name="Cameras",
        description="Number of camera viewpoints. 4 Cameras for a quick process, 16 for more details.",
        items=[
            ("4", "4 Camera Viewpoints", ""),
            ("9", "9 Camera Viewpoints", ""),
            ("16", "16 Camera Viewpoints", ""),
        ],
    )

    bpy.types.Scene.texture_seed = IntProperty(
        name="Seed",
        description="Seed for randomization to ensure repeatable results",
        default=0,
        min=0,
    )

    # bpy.types.Scene.checkpoint_path = StringProperty(
    #     name="Checkpoint Path",
    #     description="Optional path to the Stable Diffusion base model checkpoint",
    #     subtype="FILE_PATH",
    # )

    bpy.types.Scene.num_loras = IntProperty(
        name="Number of LoRAs",
        description="Number of additional LoRA models to use",
        default=0,
        min=0,
        update=update_loras,
    )

    bpy.types.Scene.lora_models = CollectionProperty(type=LoRAModel)

    # IPAdapter-specific properties
    bpy.types.Scene.use_ipadapter = bpy.props.BoolProperty(
        name="Use IPAdapter",
        description="Activate IPAdapter for texture generation",
        default=False,
    )

    bpy.types.Scene.ipadapter_image = bpy.props.PointerProperty(
        type=bpy.types.Image,
        name="IPAdapter Image",
        description="Select an image to use for IPAdapter",
        update=update_ipadapter_image,  # Attach the update callback
    )

    bpy.types.Scene.ipadapter_strength = bpy.props.FloatProperty(
        name="IPAdapter Strength",
        description="This method controls the amount of text or image conditioning to apply to the model. A value of 1.0 means the model is only conditioned on the image prompt. Lowering this value encourages the model to produce more diverse images, but they may not be as aligned with the image prompt. Typically, a value of 0.5 achieves a good balance between the two prompt types and produces good results.",
        default=0.5,
        min=0.0,
        soft_max=1.0,
    )

    # bpy.types.Scene.show_advanced = bpy.props.BoolProperty(
    #     name="Show Advanced Settings",
    #     description="Toggle the visibility of advanced settings",
    #     default=False,
    # )
    # bpy.types.Scene.canny_controlnet_strength = bpy.props.FloatProperty(
    #     name="Canny ControlNet Strength",
    #     description="Strength of the Canny ControlNet",
    #     default=1.0,
    #     min=0.0,
    #     max=2.0,
    # )
    # bpy.types.Scene.normal_controlnet_strength = bpy.props.FloatProperty(
    #     name="Normal ControlNet Strength",
    #     description="Strength of the Normal ControlNet",
    #     default=1.0,
    #     min=0.0,
    #     max=2.0,
    # )
    # bpy.types.Scene.depth_controlnet_strength = bpy.props.FloatProperty(
    #     name="Depth ControlNet Strength",
    #     description="Strength of the Depth ControlNet",
    #     default=1.0,
    #     min=0.0,
    #     max=2.0,
    # )


def unregister_properties():

    bpy.utils.unregister_class(LoRAModel)
    del bpy.types.Scene.num_loras
    del bpy.types.Scene.lora_models
    del bpy.types.Scene.use_ipadapter
    del bpy.types.Scene.ipadapter_image
    del bpy.types.Scene.ipadapter_strength
    del bpy.types.Scene.my_mesh_object
    del bpy.types.Scene.my_uv_map
    del bpy.types.Scene.my_prompt
    del bpy.types.Scene.my_negative_prompt
    del bpy.types.Scene.guidance_scale
    del bpy.types.Scene.mesh_complexity
    del bpy.types.Scene.texture_resolution
    del bpy.types.Scene.render_resolution
    del bpy.types.Scene.operation_mode
    del bpy.types.Scene.denoise_strength
    del bpy.types.Scene.output_path
    del bpy.types.Scene.texture_seed
    # del bpy.types.Scene.checkpoint_path
    # del bpy.types.Scene.show_advanced
    # del bpy.types.Scene.canny_controlnet_strength
    # del bpy.types.Scene.normal_controlnet_strength
    # del bpy.types.Scene.depth_controlnet_strength
