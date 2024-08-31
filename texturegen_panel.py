import sys
import os
import bpy
import time
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    CollectionProperty,
    EnumProperty,
    PointerProperty,
)

# Manually specify the directory containing your scripts
script_dir = "C:/Users/fredd/Desktop/SD-texturing/code/texturegen_addon"

# Add the directory to the system path if not already present
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now you can import your modules
from scene_backup import SceneBackup, clean_scene
from object_ops import move_object_to_origin, calculate_mesh_midpoint
from texturegen import first_pass, second_pass, uv_pass


class LoRAModel(bpy.types.PropertyGroup):
    path: StringProperty(
        name="LoRA Path", description="Path to the LoRA model file", subtype="FILE_PATH"
    )
    strength: FloatProperty(
        name="Strength",
        description="Strength of the LoRA model",
        default=1.0,
        min=0.0,
        max=1.0,
    )


def update_loras(self, context):
    scene = context.scene
    num_loras = scene.num_loras
    lora_models = scene.lora_models

    while len(lora_models) < num_loras:
        lora_models.add()

    while len(lora_models) > num_loras:
        lora_models.remove(len(lora_models) - 1)


def update_image_list(self, context):
    """Update the list of available images for the selected UV map."""
    images = [(img.name, img.name, "") for img in bpy.data.images]
    if not images:
        images.append(("None", "None", "No images available"))
    return images


class OBJECT_OT_GenerateTexture(bpy.types.Operator):
    bl_idname = "object.generate_texture"
    bl_label = "Generate Texture"

    def execute(self, context):
        scene = context.scene
        output_path = scene.output_path
        selected_mesh_name = scene.my_mesh_object
        selected_object = bpy.data.objects.get(selected_mesh_name)

        # Ensure the output path exists
        if not output_path:
            self.report({"ERROR"}, "Output path is not set.")
            return {"CANCELLED"}

        if not os.path.exists(output_path):
            self.report({"ERROR"}, "Output path does not exist.")
            return {"CANCELLED"}

        if not any([scene.first_pass, scene.second_pass, scene.refinement_uv_space]):
            self.report({"ERROR"}, "Select at least one pass.")
            return {"CANCELLED"}

        # Save a backup of the current .blend file
        backup_file = os.path.join(output_path, "scene_backup.blend")
        bpy.ops.wm.save_as_mainfile(filepath=backup_file)

        try:
            # Start progress indicator using context.window_manager
            wm = context.window_manager
            wm.progress_begin(0, 100)

            # Clean the scene, removing all other objects
            clean_scene(scene)

            # Move object to world origin and calculate midpoint
            max_size = calculate_mesh_midpoint(selected_object)
            scene.radius = max_size
            move_object_to_origin(selected_object)

            # If an image is selected, apply it to the UV map
            if scene.selected_image:
                image = bpy.data.images.get(scene.selected_image)
                if image:
                    for uv_layer in selected_object.data.uv_layers:
                        if uv_layer.name == scene.my_uv_map:
                            # Assuming you're using the correct way to assign the image to the UV map
                            for tex_slot in selected_object.material_slots[
                                0
                            ].material.texture_paint_slots:
                                tex_slot.texture.image = image

            # Execute texture passes based on user selection
            if scene.first_pass:
                first_pass.first_pass(scene)
            if scene.second_pass:
                second_pass.second_pass(scene)
            if scene.refinement_uv_space:
                uv_pass.uv_pass(scene)

            # Process complete
            wm.progress_end()

        except Exception as e:
            self.report({"ERROR"}, f"An error occurred: {e}")
            wm.progress_end()
            return {"CANCELLED"}

        # Restore the original scene by reloading the backup file
        bpy.ops.wm.open_mainfile(filepath=backup_file)

        return {"FINISHED"}


class OBJECT_PT_CustomPanel(bpy.types.Panel):
    bl_label = "Stable Diffuse Texture Diffusion"
    bl_idname = "OBJECT_PT_custom_panel"
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

        # Image Selection (Optional)
        layout.prop(scene, "selected_image", text="Image")

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

        # First Pass in Image Space Checkbox
        layout.prop(scene, "first_pass", text="First Pass from Gaussian Noise")

        # Second Pass in Image Space Checkbox
        layout.prop(scene, "second_pass", text="Second Pass in Image Space (img2img)")

        # Refinement in UV Space Checkbox
        layout.prop(
            scene, "refinement_uv_space", text="Refinement in UV Space (img2img)"
        )

        # Seed Input Field
        layout.prop(scene, "texture_seed", text="Seed")

        # Optional: Checkpoint Selection
        layout.prop(scene, "checkpoint_path", text="Checkpoint")

        # LoRA models
        layout.prop(scene, "num_loras", text="Number of LoRAs")

        for i in range(scene.num_loras):
            box = layout.box()
            box.label(text=f"LoRA Model {i+1}")
            lora = scene.lora_models[i]
            box.prop(lora, "path", text="Path LoRA")
            box.prop(lora, "strength", text="Strength LoRA")

        # Big Button for Texture Generation
        row = layout.row()
        row.scale_y = 2.0  # Makes the button bigger
        row.operator(
            "object.generate_texture",
            text="Start Texture Generation",
            icon="SHADERFX",
        )


class OBJECT_OT_SelectPipette(bpy.types.Operator):
    """Set the selected object in the 3D view as the mesh object in the panel"""

    bl_idname = "object.select_pipette"
    bl_label = "Select Object with Pipette"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.my_mesh_object = context.object.name
        return {"FINISHED"}


def update_uv_maps(self, context):
    obj = bpy.data.objects.get(self.my_mesh_object)
    if obj and obj.type == "MESH":
        uv_layers = obj.data.uv_layers.keys()
        return [(uv, uv, "") for uv in uv_layers]
    else:
        return [("None", "None", "")]


def get_mesh_objects(self, context):
    return [(obj.name, obj.name, "") for obj in bpy.data.objects if obj.type == "MESH"]


def register():
    bpy.utils.register_class(LoRAModel)
    bpy.utils.register_class(OBJECT_OT_GenerateTexture)
    bpy.utils.register_class(OBJECT_OT_SelectPipette)
    bpy.utils.register_class(OBJECT_PT_CustomPanel)

    bpy.types.Scene.my_mesh_object = bpy.props.EnumProperty(
        name="Mesh Object",
        items=get_mesh_objects,
        description="Select the mesh object you want to use texturize.",
    )

    bpy.types.Scene.my_uv_map = bpy.props.EnumProperty(
        name="UV Map",
        items=update_uv_maps,
        description="Select the UV map you want to use for the final texture.",
    )

    bpy.types.Scene.selected_image = EnumProperty(
        name="Image",
        description="Select an image to use with the UV Map",
        items=update_image_list,
    )

    bpy.types.Scene.my_prompt = bpy.props.StringProperty(
        name="Prompt", description="Define what the object should be"
    )

    bpy.types.Scene.my_negative_prompt = bpy.props.StringProperty(
        name="Negative Prompt", description="Define what the object should NOT be"
    )

    bpy.types.Scene.mesh_complexity = bpy.props.EnumProperty(
        name="Mesh Complexity",
        description="The complexity, polycount and detail of the selected mesh.",
        items=[("LOW", "Low", ""), ("MEDIUM", "Medium", ""), ("HIGH", "High", "")],
    )

    bpy.types.Scene.texture_resolution = bpy.props.EnumProperty(
        name="Texture Resolution",
        description="The final texture resolution of the selected mesh object.",
        items=[
            ("512", "512x512", ""),
            ("1024", "1024x1024", ""),
            ("2048", "2048x2048", ""),
            ("4096", "4096x4096", ""),
        ],
    )

    bpy.types.Scene.output_path = bpy.props.StringProperty(
        name="Output Path",
        description="Directory to store the resulting texture and temporary files",
        subtype="DIR_PATH",
        default=os.path.dirname(bpy.data.filepath) if bpy.data.filepath else "",
    )

    bpy.types.Scene.first_pass = bpy.props.BoolProperty(
        name="First Pass from Gaussian Noise",
        description="First pass creates one result from multiple viewpoints at once.",
    )

    bpy.types.Scene.second_pass = bpy.props.BoolProperty(
        name="Second Pass in Image Space",
        description="Second pass improves results from multiple viewpoints successively.",
    )

    bpy.types.Scene.refinement_uv_space = bpy.props.BoolProperty(
        name="Refinement in UV Space",
        description="Refinement in UV space fixes areas not visible to the camera or holes in the texture.",
    )

    bpy.types.Scene.texture_seed = bpy.props.IntProperty(
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


def unregister():
    bpy.utils.unregister_class(LoRAModel)
    bpy.utils.unregister_class(OBJECT_OT_GenerateTexture)
    bpy.utils.unregister_class(OBJECT_OT_SelectPipette)
    bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
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
    del bpy.types.Scene.num_loras
    del bpy.types.Scene.lora_models


if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
