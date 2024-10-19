import os
import bpy
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    CollectionProperty,
    EnumProperty,
    BoolProperty,
    PointerProperty,
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
    if self.operation_mode == "TEXT2IMAGE_PARALLEL":
        self.denoise_strength = 1.0  # Set denoise strength to 1.0
    elif self.operation_mode == "TEXTURE2TEXTURE_ENHANCEMENT":
        self.denoise_strength = 0.1  # Set denoise strength to 1.0
    else:
        self.denoise_strength = (
            0.4  # Set default to 0.4 if switching away from TEXT2IMAGE_PARALLEL
        )


def update_ipadapter_image(self, context):
    """Ensure the selected image from the preview window is set in scene.ipadapter_image."""
    image = context.scene.ipadapter_image
    if image:
        image_data = bpy.data.images.get(image.name)

        # Only set the image if it's not already correctly set to prevent recursion
        if image_data != context.scene.ipadapter_image:
            context.scene.ipadapter_image = image_data


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
            (
                "TEXTURE2TEXTURE_ENHANCEMENT",
                "Texture2Texture Enhancement",
                "Enhance textures using input textures.",
            ),
        ],
        update=update_operation_mode,  # Add the update function
    )

    bpy.types.Scene.denoise_strength = FloatProperty(
        name="Denoise Strength",
        description="Strength of denoise for Stable Diffusion",
        default=0.4,
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

    bpy.types.Scene.mesh_complexity = EnumProperty(
        name="Mesh Complexity",
        description="How complex is the mesh.",
        items=[("LOW", "Low", ""), ("MEDIUM", "Medium", ""), ("HIGH", "High", "")],
    )

    bpy.types.Scene.num_cameras = EnumProperty(
        name="Number of Cameras",
        description="How many viewpoints are to be used?",
        items=[
            ("4", "4 Camera Viewpoints", ""),
            ("9", "9 Camera Viewpoints", ""),
            ("16", "16 Camera Viewpoints", ""),
        ],
    )

    # bpy.types.Scene.first_pass = BoolProperty(
    #     name="First Pass from Gaussian Noise",
    #     description="First pass creates one result from multiple viewpoints at once.",
    # )

    # bpy.types.Scene.num_cameras_first_pass = EnumProperty(
    #     name="Number of Cameras (first pass)",
    #     description="How many viewpoints are to be used to create the initial texture?",
    #     items=[
    #         ("4", "4 Camera Viewpoints", ""),
    #         ("9", "9 Camera Viewpoints", ""),
    #         ("16", "16 Camera Viewpoints", ""),
    #     ],
    # )

    # bpy.types.Scene.second_pass = BoolProperty(
    #     name="Second Pass in Image Space",
    #     description="Second pass improves results from multiple viewpoints at once.",
    # )

    # bpy.types.Scene.num_cameras_second_pass = EnumProperty(
    #     name="Number of Cameras (second pass)",
    #     description="How many viewpoints are to be used to refine the initial texture?",
    #     items=[
    #         ("4", "4 Camera Viewpoints", ""),
    #         ("9", "9 Camera Viewpoints", ""),
    #         ("16", "16 Camera Viewpoints", ""),
    #     ],
    # )

    # bpy.types.Scene.third_pass = BoolProperty(
    #     name="Third Pass in Image Space",
    #     description="Third pass improves results from multiple viewpoints successively for details.",
    # )

    # bpy.types.Scene.refinement_uv_space = BoolProperty(
    #     name="Refinement in UV Space",
    #     description="Refinement in UV space fixes areas not visible to the camera or holes in the texture.",
    # )

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


def unregister_properties():

    try:
        bpy.utils.unregister_class(LoRAModel)
    except Exception as e:
        print(
            f"Warning: {LoRAModel.__name__} was not registered or failed to unregister. {e}"
        )

    # Safely unregister properties with error handling
    try:
        del bpy.types.Scene.num_loras
    except AttributeError:
        print("num_loras not found, skipping...")

    try:
        del bpy.types.Scene.lora_models
    except AttributeError:
        print("lora_models not found, skipping...")

    try:
        del bpy.types.Scene.use_ipadapter
    except AttributeError:
        print("use_ipadapter not found, skipping...")

    try:
        del bpy.types.Scene.ipadapter_image
    except AttributeError:
        print("ipadapter_image not found, skipping...")

    try:
        del bpy.types.Scene.ipadapter_strength
    except AttributeError:
        print("ipadapter_strength not found, skipping...")

    try:
        del bpy.types.Scene.my_mesh_object
    except AttributeError:
        print("my_mesh_object not found, skipping...")

    try:
        del bpy.types.Scene.my_uv_map
    except AttributeError:
        print("my_uv_map not found, skipping...")

    try:
        del bpy.types.Scene.selected_image
    except AttributeError:
        print("selected_image not found, skipping...")

    try:
        del bpy.types.Scene.my_prompt
    except AttributeError:
        print("my_prompt not found, skipping...")

    try:
        del bpy.types.Scene.my_negative_prompt
    except AttributeError:
        print("my_negative_prompt not found, skipping...")

    try:
        del bpy.types.Scene.mesh_complexity
    except AttributeError:
        print("mesh_complexity not found, skipping...")

    try:
        del bpy.types.Scene.texture_resolution
    except AttributeError:
        print("texture_resolution not found, skipping...")

    try:
        del bpy.types.Scene.operation_mode
    except AttributeError:
        print("operation_mode not found, skipping...")

    try:
        del bpy.types.Scene.denoise_strength
    except AttributeError:
        print("denoise_strength not found, skipping...")

    try:
        del bpy.types.Scene.output_path
    except AttributeError:
        print("output_path not found, skipping...")

    # try:
    #     del bpy.types.Scene.first_pass
    # except AttributeError:
    #     print("first_pass not found, skipping...")

    # try:
    #     del bpy.types.Scene.second_pass
    # except AttributeError:
    #     print("second_pass not found, skipping...")

    # try:
    #     del bpy.types.Scene.third_pass
    # except AttributeError:
    #     print("third_pass not found, skipping...")

    # try:
    #     del bpy.types.Scene.refinement_uv_space
    # except AttributeError:
    #     print("refinement_uv_space not found, skipping...")

    try:
        del bpy.types.Scene.texture_seed
    except AttributeError:
        print("texture_seed not found, skipping...")

    # try:
    #     del bpy.types.Scene.checkpoint_path
    # except AttributeError:
    #     print("checkpoint_path not found, skipping...")

    print("Properties unregistered successfully.")
