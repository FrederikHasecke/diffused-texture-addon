import bpy


class OBJECT_PT_IPAdapterPanel(bpy.types.Panel):
    """IpAdapter Panel.

    Args:
        bpy (_type_): _description_
    """

    bl_label = "IPAdapter"
    bl_idname = "OBJECT_PT_ipadapter_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw(self, context: bpy.types.Context) -> None:
        """Draw function.

        Args:
            context (bpy.types.Context): _description_
        """
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
    """Operator to open a new image for IPAdapter."""

    bl_idname = "image.open_new_ipadapter_image"
    bl_label = "Open New IPAdapter Image"
    bl_order = 3

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore  # noqa: PGH003

    def execute(
        self: "OBJECT_OT_OpenNewIPAdapterImage",
        context: bpy.types.Context,
    ) -> set[str]:
        """Execute.

        Args:
            context (bpy.types.Context): _description_

        Returns:
            set[str]: _description_
        """
        # Load the new image using the provided filepath
        image = bpy.data.images.load(self.filepath)
        context.scene.ipadapter_image = image
        return {"FINISHED"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:  # noqa: ARG002
        """Invoke the file selection dialog."""
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}
