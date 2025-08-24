import bpy


class OBJECT_OT_SelectPipette(bpy.types.Operator):
    """Operator to select and object."""

    bl_idname = "object.select_pipette"
    bl_label = "Select Object with Pipette"

    def execute(
        self: "OBJECT_OT_SelectPipette",
        context: bpy.types.Context,
    ) -> set[str]:
        """.

        Args:
            context (bpy.types.Context): _description_

        Returns:
            set[str]: _description_
        """
        context.scene.my_mesh_object = context.object.name
        return {"FINISHED"}


class OBJECT_OT_OpenNewInputImage(bpy.types.Operator):
    """Operator to open a new image for the input texture."""

    bl_idname = "image.open_new_input_image"
    bl_label = "Open New Input Image"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore  # noqa: PGH003

    def execute(
        self: "OBJECT_OT_OpenNewInputImage",
        context: bpy.types.Context,
    ) -> set[str]:
        """Execute the operator to load a new input image."""
        # Load the new image using the provided filepath
        image = bpy.data.images.load(self.filepath)
        context.scene.input_texture = image
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:  # noqa: ARG002
        """Select the file.

        Args:
            context (bpy.types.Context): _description_
            event (_type_): _description_

        Returns:
            set[str]: _description_
        """
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class OBJECT_PT_DiffusedTextureMainPanel(bpy.types.Panel):
    """Main Panel.

    Args:
        bpy (_type_): _description_
    """

    bl_label = "DiffusedTexture"
    bl_idname = "OBJECT_PT_diffused_texture_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_order = 0

    def draw(self, context: bpy.types.Context) -> None:  # noqa: PLR0915
        """Draw Function.

        Args:
            context (bpy.context.scene): _description_
        """
        layout = self.layout
        scene = context.scene

        # Object and UV selection
        row = layout.row(align=True)
        row.prop(scene, "my_mesh_object", text="Mesh Object")
        row.operator("object.select_pipette", text="", icon="VIS_SEL_11")
        layout.prop(scene, "my_uv_map", text="UV Map")

        # Stable Diffusion configuration
        box_sd = layout.box()
        box_sd.label(text="Stable Diffusion Settings")
        box_sd.prop(scene, "my_prompt", text="Prompt")
        box_sd.prop(scene, "my_negative_prompt", text="Negative Prompt")
        box_sd.prop(scene, "denoise_strength", text="Denoise Strength")
        box_sd.prop(scene, "num_inference_steps", text="Inference Steps")
        box_sd.prop(scene, "guidance_scale", text="Guidance Scale")

        # DiffusedTexture settings
        box_dt = layout.box()
        box_dt.label(text="Texture Generation Settings")
        box_dt.prop(scene, "operation_mode", text="Operation Mode")

        if scene.operation_mode == "PARA_SEQUENTIAL_IMG":
            box_dt.prop(scene, "subgrid_rows", text="Subgrid Rows")
            box_dt.prop(scene, "subgrid_cols", text="Subgrid Columns")

        box_dt.prop(scene, "mesh_complexity", text="Mesh Complexity")
        box_dt.prop(scene, "num_cameras", text="Camera Views")

        if scene.num_cameras == "16":
            box_dt.label(
                text="Warning: 16 cameras may freeze Blender or cause OOM",
                icon="ERROR",
            )

        box_dt.prop(scene, "texture_resolution", text="Texture Resolution")
        box_dt.prop(scene, "render_resolution", text="Render Resolution")

        if int(scene.render_resolution) <= int(scene.texture_resolution):
            box_dt.label(
                text="Render resolution should be at least 2x texture resolution",
                icon="ERROR",
            )

        box_dt.prop(scene, "output_path", text="Output Path")
        if scene.output_path.startswith("//"):
            box_dt.label(
                text=f"Absolute Path: {bpy.path.abspath(scene.output_path)}",
                icon="FILE_FOLDER",
            )
        elif not scene.output_path:
            box_dt.label(text="Warning: No output path given!", icon="ERROR")

        box_dt.prop(scene, "texture_seed", text="Seed")

        # Input Texture
        layout.label(text="Input Texture")
        row = layout.row()
        row.template_ID_preview(scene, "input_texture", rows=2, cols=6)
        layout.operator(
            "image.open_new_input_image",
            text="Open Input Texture",
            icon="IMAGE_DATA",
        )

        # Texture Generation Button
        if not scene.output_path.strip():
            layout.label(
                text="Please set output path to start generation",
                icon="ERROR",
            )
            row = layout.row()
            row.scale_y = 2.0
            row.enabled = False
            row.operator(
                "object.generate_texture",
                text="Start Texture Generation",
                icon="SHADERFX",
            )
        else:
            if not bpy.app.online_access:
                box = layout.box()
                box.label(text="Online access is disabled", icon="ERROR")
                box.label(text="Enable it in Preferences > System > Network")
            row = layout.row()
            row.scale_y = 2.0
            row.operator(
                "object.generate_texture",
                text="Start Texture Generation",
                icon="SHADERFX",
            )
