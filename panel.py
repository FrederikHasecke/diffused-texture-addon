import os
import sys
import bpy

############################################################################
# TODO: REMOVE THIS AFTER ALL WORKS!!
# Manually specify the directory containing your scripts
script_dir = "C:/Users/fredd/Desktop/SD-texturing/code/texturegen_addon"
# Add the directory to the system path if not already present
if script_dir not in sys.path:
    sys.path.append(script_dir)

os.environ["HF_HOME"] = r"G:\Huggingface_cache"
############################################################################

from operators import OBJECT_OT_GenerateTexture, OBJECT_OT_SelectPipette
from properties import register_properties, unregister_properties


class OBJECT_PT_MainPanel(bpy.types.Panel):
    bl_label = "Stable Diffuse Texture Diffusion"
    bl_idname = "OBJECT_PT_texturegen_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TextureGen"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Object Selection
        row = layout.row(align=True)
        row.prop(scene, "my_mesh_object", text="Mesh Object")
        row.operator("object.select_pipette", text="", icon="VIS_SEL_11")

        # UV Map Selection
        layout.prop(scene, "my_uv_map", text="UV Map")

        # Prompt Text Field
        layout.prop(scene, "my_prompt", text="Prompt")

        # Negative Prompt Text Field
        layout.prop(scene, "my_negative_prompt", text="Negative Prompt")

        # Mesh Complexity Dropdown
        layout.prop(scene, "mesh_complexity", text="Mesh Complexity")

        # Texture Resolution Dropdown
        layout.prop(scene, "texture_resolution", text="Texture Resolution")

        # Warning for High Resolution
        if scene.texture_resolution in ["1024", "2048", "4096"]:
            layout.label(
                text="All three passes required for >=1024 resolution",
                icon="ERROR",
            )

        # Output Directory Path
        layout.prop(scene, "output_path", text="Output Path")

        # Warning for missing texture
        if scene.output_path == "":
            layout.label(
                text="Warning: No Output Path Given!",
                icon="ERROR",
            )

        # Input Texture Path (for img2img or texture2texture pass)
        layout.prop(
            scene,
            "input_texture_path",
            text="Input Texture (for img2img or texture2texture pass)",
        )

        # # First Pass in Image Space Checkbox
        # layout.prop(
        #     scene, "first_pass", text="First Pass from Gaussian Noise (text2img)"
        # )

        # operation_mode Dropdown
        layout.prop(scene, "operation_mode", text="Operation Mode")

        # Denoise
        layout.prop(scene, "denoise_strength", text="Denoise Strength")

        # Num Cameras Dropdow
        layout.prop(
            scene,
            "num_cameras",
            text="Number of camera viewpoints. 4 Cameras for a quick process, 16 for more details.",
        )

        # Warning for Many Cameras
        if scene.num_cameras == "16":
            layout.label(
                text="Warning: High VRAM, ~5min freeze",
                icon="ERROR",
            )

        # Seed Input Field
        layout.prop(scene, "texture_seed", text="Seed")

        # # Optional: Checkpoint Selection
        # layout.prop(scene, "checkpoint_path", text="Checkpoint")

        # Execute Button
        row = layout.row()
        row.scale_y = 2.0  # Makes the button bigger
        row.operator(
            "object.generate_texture",
            text="Start Texture Generation",
            icon="SHADERFX",
        )


class OBJECT_PT_LoRAPanel(bpy.types.Panel):
    bl_label = "LoRA Models"
    bl_idname = "OBJECT_PT_lora_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TextureGen"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "num_loras", text="Number of LoRAs")

        for i in range(scene.num_loras):
            lora_box = layout.box()
            lora_box.label(text=f"LoRA Model {i+1}")
            lora = scene.lora_models[i]
            lora_box.prop(lora, "path", text="Path LoRA")
            lora_box.prop(lora, "strength", text="Strength LoRA")


class OBJECT_PT_IPAdapterPanel(bpy.types.Panel):
    bl_label = "IPAdapter"
    bl_idname = "OBJECT_PT_ipadapter_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TextureGen"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # IPAdapter Activation Checkbox
        layout.prop(scene, "use_ipadapter", text="Activate IPAdapter")

        # IPAdapter Image Preview and Selection
        row = layout.row()
        row.template_ID_preview(scene, "ipadapter_image", rows=2, cols=6)

        # Button to open the file browser and load a new image
        layout.operator(
            "image.open_new_ipadapter_image", text="Open New Image", icon="IMAGE_DATA"
        )

        # IPAdapter Strength Slider
        layout.prop(scene, "ipadapter_strength", text="Strength IPAdapter")


class OBJECT_OT_OpenNewIPAdapterImage(bpy.types.Operator):
    """Operator to open a new image for IPAdapter"""

    bl_idname = "image.open_new_ipadapter_image"
    bl_label = "Open New IPAdapter Image"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        # Load the new image using the provided filepath
        image = bpy.data.images.load(self.filepath)
        context.scene.ipadapter_image = image
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def register():
    # Unregister everything first to avoid duplication
    try:
        unregister_properties()
    except Exception as e:
        print(f"Error during property unregistration: {e}")

    # Unregister all the classes if already registered
    classes = [
        OBJECT_OT_GenerateTexture,
        OBJECT_OT_SelectPipette,
        OBJECT_PT_MainPanel,
        OBJECT_PT_LoRAPanel,
        OBJECT_PT_IPAdapterPanel,
        OBJECT_OT_OpenNewIPAdapterImage,
    ]

    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Error unregistering {cls.__name__}: {e}")

    # Now re-register everything
    try:
        register_properties()
        for cls in classes:
            bpy.utils.register_class(cls)
    except Exception as e:
        print(f"Error during registration: {e}")


def unregister():
    # Unregister all classes
    classes = [
        OBJECT_OT_GenerateTexture,
        OBJECT_OT_SelectPipette,
        OBJECT_PT_MainPanel,
        OBJECT_PT_LoRAPanel,
        OBJECT_PT_IPAdapterPanel,
        OBJECT_OT_OpenNewIPAdapterImage,
    ]

    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Error unregistering {cls.__name__}: {e}")

    try:
        unregister_properties()
    except Exception as e:
        print(f"Error during property unregistration: {e}")


if __name__ == "__main__":
    try:
        unregister()  # Unregister first to ensure no conflicts
    except Exception as e:
        print(f"Unregister error: {e}")

    register()  # Then register all
