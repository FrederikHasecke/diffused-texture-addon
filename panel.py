import bpy


class OBJECT_PT_MainPanel(bpy.types.Panel):
    bl_label = "DiffusedTexture"
    bl_idname = "OBJECT_PT_diffused_texture_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_order = 0  # Main panel order

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

        # Num Inference Steps
        box_sd.prop(scene, "num_inference_steps", text="Number of Inference Steps")

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

        # Warning for missing output path
        if scene.output_path == "":
            box_dt.label(
                text="Warning: No Output Path Given!",
                icon="ERROR",
            )

        # Seed Input Field
        box_dt.prop(scene, "texture_seed", text="Seed")

        # Text for the user to select the input texture
        layout.label(text="Input Texture")

        # Input Image Preview and Selection
        row = layout.row()
        row.template_ID_preview(scene, "input_texture_path", rows=2, cols=6)

        # Button to open the file browser and load a new image
        layout.operator(
            "image.open_new_input_image",
            text="Open New Input Texture",
            icon="IMAGE_DATA",
        )

        # Button to execute the texture generation function

        # Disable the button if the output path is not specified
        if not bool(scene.output_path.strip()):
            layout.label(
                text="Please specify an output path to start texture generation.",
                icon="ERROR",
            )
            row = layout.row()
            row.scale_y = 2.0
            row.operator(
                "object.generate_texture",
                text="Start Texture Generation",
                icon="SHADERFX",
            )
            row.enabled = False
        else:
            # Enable the button if the output path is specified but give a warning
            if not bpy.app.online_access:
                box = layout.box()
                box.label(text="Online access is disabled.", icon="ERROR")
                box.label(
                    text="If you don't have the models installed, please enable it in"
                )
                box.label(text="Preferences > System > Network > Allow Online Access,")
                box.label(text="else the texture generation will fail.")
                row = layout.row()
                row.scale_y = 2.0
                row.operator(
                    "object.generate_texture",
                    text="Start Texture Generation",
                    icon="SHADERFX",
                )
            else:
                # Enable the button if the output path is specified
                row = layout.row()
                row.scale_y = 2.0
                row.operator(
                    "object.generate_texture",
                    text="Start Texture Generation",
                    icon="SHADERFX",
                )


class OBJECT_OT_OpenNewInputImage(bpy.types.Operator):
    """Operator to open a new image for the input texture"""

    bl_idname = "image.open_new_input_image"
    bl_label = "Open New Input Image"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        # Load the new image using the provided filepath
        image = bpy.data.images.load(self.filepath)
        context.scene.input_texture_path = image
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class OBJECT_PT_LoRAPanel(bpy.types.Panel):
    bl_label = "LoRA Models"
    bl_idname = "OBJECT_PT_lora_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "num_loras", text="Number of LoRAs")

        for i in range(scene.num_loras):
            lora_box = layout.box()
            lora_box.label(text=f"LoRA Model {i+1}")
            lora = scene.lora_models[i]
            lora_box.prop(lora, "path", text="Path LoRA")

            if lora.path.startswith("//"):
                absolute_path = bpy.path.abspath(lora.path)
                lora_box.label(
                    text=f"Absolute Path: {absolute_path}", icon="FILE_FOLDER"
                )

            lora_box.prop(lora, "strength", text="Strength LoRA")


class OBJECT_PT_IPAdapterPanel(bpy.types.Panel):
    bl_label = "IPAdapter"
    bl_idname = "OBJECT_PT_ipadapter_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # IPAdapter Activation Checkbox
        layout.prop(scene, "use_ipadapter", text="Activate IPAdapter")

        # IPAdapter Image Preview and Selection
        row = layout.row()

        # disable the IPAdapter image selection if the IPAdapter is not activated
        row.enabled = scene.use_ipadapter
        row.template_ID_preview(scene, "ipadapter_image", rows=2, cols=6)

        row = layout.row()
        row.enabled = scene.use_ipadapter
        # Button to open the file browser and load a new image
        row.operator(
            "image.open_new_ipadapter_image",
            text="Open New IPAdapter Image",
            icon="IMAGE_DATA",
        )

        # IPAdapter Strength Slider
        row = layout.row()
        row.enabled = scene.use_ipadapter
        row.prop(scene, "ipadapter_strength", text="Strength IPAdapter")


class OBJECT_OT_OpenNewIPAdapterImage(bpy.types.Operator):
    """Operator to open a new image for IPAdapter"""

    bl_idname = "image.open_new_ipadapter_image"
    bl_label = "Open New IPAdapter Image"
    bl_order = 3

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        # Load the new image using the provided filepath
        image = bpy.data.images.load(self.filepath)
        context.scene.ipadapter_image = image
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class OBJECT_PT_AdvancedPanel(bpy.types.Panel):
    """Advanced Settings Panel"""

    bl_label = "Advanced Settings"
    bl_idname = "OBJECT_PT_advanced_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()

        # dropdown menu for the sd model (sd15 or sdxl so far)
        box.prop(scene, "sd_version", text="Stable Diffusion Version:")

        # custom SD checkpoints
        box.prop(scene, "checkpoint_path", text="Checkpoint")

        # custom SD resolution
        box.prop(scene, "custom_sd_resolution", text="Custom SD Resolution")

        # warning for the user to not go too high with the resolution, especially for the parallel operation,
        # since the resolution will be multiplied by sqrt(num of cameras)
        if scene.custom_sd_resolution:
            box.label(
                text="Warning: High resolutions can lead to memory issues and long processing times.",
                icon="ERROR",
            )
            box.label(
                text="The resolution will be multiplied by the square root of the number of cameras.",
            )
            box.label(
                text="It is recommended to keep the resolution below 1024.",
            )

        # Add advanced settings
        box.label(text="ControlNet Checkpoints:")
        box.prop(scene, "depth_controlnet_path", text="Depth Path")

        # disable the canny_controlnet_path option if the mesh complexity is set to LOW
        row = box.row()
        row.enabled = scene.mesh_complexity != "LOW"
        row.prop(scene, "canny_controlnet_path", text="Canny Path")

        # disable the normal_controlnet_path option if the mesh complexity is set to LOW or MID
        row = box.row()
        row.enabled = scene.mesh_complexity == "HIGH"
        row.prop(scene, "normal_controlnet_path", text="Normal Path")

        box.label(text="ControlNet Strengths:")
        box.prop(scene, "depth_controlnet_strength", text="Depth Strength")

        # disable the canny_controlnet_strength option if the mesh complexity is set to LOW
        row = box.row()
        row.enabled = scene.mesh_complexity != "LOW"
        row.prop(scene, "canny_controlnet_strength", text="Canny Strength")

        # disable the normal_controlnet_strength option if the mesh complexity is set to LOW or MID
        row = box.row()
        row.enabled = scene.mesh_complexity == "HIGH"
        row.prop(scene, "normal_controlnet_strength", text="Normal Strength")
