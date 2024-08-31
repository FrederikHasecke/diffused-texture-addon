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
############################################################################

from operators import OBJECT_OT_GenerateTexture, OBJECT_OT_SelectPipette
from properties import register_properties, unregister_properties


class OBJECT_PT_MainPanel(bpy.types.Panel):
    bl_label = "Stable Diffuse Texture Diffusion"
    bl_idname = "shade.texturegen"
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
        if scene.texture_resolution == "4096":
            layout.label(
                text="Warning: High VRAM required for 4096x4096 resolution!",
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

        # First Pass in Image Space Checkbox
        layout.prop(
            scene, "first_pass", text="First Pass from Gaussian Noise (text2img)"
        )

        # Second Pass in Image Space Checkbox
        layout.prop(scene, "second_pass", text="Second Pass in Image Space (img2img)")

        # Refinement in UV Space Checkbox
        layout.prop(
            scene,
            "refinement_uv_space",
            text="Refinement in UV Space (texture2texture)",
        )

        if not any([scene.first_pass, scene.second_pass, scene.refinement_uv_space]):
            layout.label(
                text="Warning: Select at Least One Pass!",
                icon="ERROR",
            )

        # Seed Input Field
        layout.prop(scene, "texture_seed", text="Seed")

        # Optional: Checkpoint Selection
        layout.prop(scene, "checkpoint_path", text="Checkpoint")

        # Execute Button
        row = layout.row()
        row.scale_y = 2.0  # Makes the button bigger
        row.operator(
            "object.generate_texture",
            text="Start Texture Generation",
            icon="SHADERFX",
        )

        # Add the button to install required libraries
        layout.operator("object.install_libraries", text="Install Required Libraries")


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

        # IPAdapter Image Path Selection
        layout.prop(scene, "ipadapter_image_path", text="Image Path IPAdapter")

        # Optional: IPAdapter Model Path Selection
        layout.prop(
            scene, "ipadapter_model_path", text="Model Path IPAdapter (Optional)"
        )

        # IPAdapter Strength Slider
        layout.prop(scene, "ipadapter_strength", text="Strength IPAdapter")


def register():
    # If libraries are available, proceed with the regular registration
    register_properties()
    bpy.utils.register_class(OBJECT_PT_MainPanel)
    bpy.utils.register_class(OBJECT_PT_LoRAPanel)
    bpy.utils.register_class(OBJECT_PT_IPAdapterPanel)
    bpy.utils.register_class(OBJECT_OT_GenerateTexture)
    bpy.utils.register_class(OBJECT_OT_SelectPipette)


def unregister():
    unregister_properties()
    bpy.utils.unregister_class(OBJECT_PT_MainPanel)
    bpy.utils.unregister_class(OBJECT_PT_LoRAPanel)
    bpy.utils.unregister_class(OBJECT_PT_IPAdapterPanel)
    bpy.utils.unregister_class(OBJECT_OT_GenerateTexture)
    bpy.utils.unregister_class(OBJECT_OT_SelectPipette)


if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
