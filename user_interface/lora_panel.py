import bpy


class OBJECT_PT_LoRAPanel(bpy.types.Panel):
    """Panel for managing LoRA models in the DiffusedTexture addon."""

    bl_label = "LoRA Models"
    bl_idname = "OBJECT_PT_lora_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context: bpy.types.Context) -> None:
        """Draw the LoRA panel in the UI."""
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "num_loras", text="Number of LoRAs")

        for i in range(scene.num_loras):
            lora_box = layout.box()
            lora_box.label(text=f"LoRA Model {i + 1}")
            lora = scene.lora_models[i]
            lora_box.prop(lora, "path", text="Path LoRA")

            if lora.path.startswith("//"):
                absolute_path = bpy.path.abspath(lora.path)
                lora_box.label(
                    text=f"Absolute Path: {absolute_path}",
                    icon="FILE_FOLDER",
                )

            lora_box.prop(lora, "strength", text="Strength LoRA")
