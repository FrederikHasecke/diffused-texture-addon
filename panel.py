import bpy


class OBJECT_PT_MainPanel(bpy.types.Panel):
    bl_label = "DiffusedTexture"
    bl_idname = "OBJECT_PT_diffused_texture_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Object Selection
        row = layout.row(align=True)
        row.prop(scene, "my_mesh_object", text="Mesh Object")
        row.operator("object.select_pipette", text="", icon="VIS_SEL_11")

        # UV Map Selection
        layout.prop(scene, "my_uv_map", text="UV Map")

        box_sd = layout.box()
        box_sd.label(text="Stable Diffusion Options")

        # Prompt Text Field
        box_sd.prop(scene, "my_prompt", text="Prompt")

        # Negative Prompt Text Field
        box_sd.prop(scene, "my_negative_prompt", text="Negative Prompt")

        # Denoise
        box_sd.prop(scene, "denoise_strength", text="Denoise Strength")

        # guidance_scale
        box_sd.prop(scene, "guidance_scale", text="Guidance Scale")

        box_dt = layout.box()
        box_dt.label(text="DiffusedTexture Options")

        # operation_mode Dropdown
        box_dt.prop(scene, "operation_mode", text="Operation Mode")

        # Mesh Complexity Dropdown
        box_dt.prop(scene, "mesh_complexity", text="Mesh Complexity")

        # Num Cameras Dropdow
        box_dt.prop(
            scene,
            "num_cameras",
            text="Cameras",
        )

        # Warning for Many Cameras
        if scene.num_cameras == "16":
            box_dt.label(
                text="Warning: long freeze, might produce OUT OF MEMORY error",
                icon="ERROR",
            )

        # Texture Resolution Dropdown
        box_dt.prop(scene, "texture_resolution", text="Texture Resolution")

        # Texture Resolution Dropdown
        box_dt.prop(scene, "render_resolution", text="Render Resolution")

        # Warning for low render Resolution
        if int(scene.render_resolution) <= int(scene.texture_resolution):
            layout.label(
                text="Render Resolution should be at least 2x Texture Resolution to prevent 'banding artifacts' in the texture",
                icon="ERROR",
            )

        # Output Directory Path
        box_dt.prop(scene, "output_path", text="Output Path")
        if scene.output_path.startswith("//"):
            absolute_path = bpy.path.abspath(scene.output_path)
            box_dt.label(text=f"Absolute Path: {absolute_path}", icon="FILE_FOLDER")

        # Warning for missing texture
        if scene.output_path == "":
            box_dt.label(
                text="Warning: No Output Path Given!",
                icon="ERROR",
            )

        # Input Texture Path (for img2img or texture2texture pass)
        box_dt.prop(
            scene,
            "input_texture_path",
            text="Input Texture",
        )

        # Seed Input Field
        box_dt.prop(scene, "texture_seed", text="Seed")

        # TODO: Checkpoint Selection which are converted to Diffusers
        # layout.prop(scene, "checkpoint_path", text="Checkpoint")

        # Execute Button
        row = layout.row()
        row.scale_y = 2.0

        # Enable the button only if the output path is specified
        row.enabled = bool(scene.output_path.strip())
        row.operator(
            "object.generate_texture",
            text="Start Texture Generation",
            icon="SHADERFX",
        )


# TODO: Test if this Lora stuff even works
class OBJECT_PT_LoRAPanel(bpy.types.Panel):
    bl_label = "LoRA Models"
    bl_idname = "OBJECT_PT_lora_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}

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
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}

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


# TODO: Give the user more options if they know the SD basics
# class OBJECT_PT_AdvancedPanel(bpy.types.Panel):
#     """Advanced Settings Panel"""
#     bl_label = "Advanced Settings"
#     bl_idname = "OBJECT_PT_advanced_panel"
#     bl_space_type = "VIEW_3D"
#     bl_region_type = "UI"
#     bl_category = "DiffuseTex"

#     def draw(self, context):
#         layout = self.layout
#         scene = context.scene

#         # Add a toggle to show/hide advanced settings
#         layout.prop(scene, "show_advanced", text="Show Advanced Settings")

#         if scene.show_advanced:
#             box = layout.box()

#             # Add advanced settings
#             box.label(text="ControlNet Strengths:")
#             box.prop(scene, "canny_controlnet_strength", text="Canny Strength")
#             box.prop(scene, "normal_controlnet_strength", text="Normal Strength")
#             box.prop(scene, "depth_controlnet_strength", text="Depth Strength")
