import math

import bpy


class OBJECT_PT_AdvancedPanel(bpy.types.Panel):
    """Advanced Settings Panel."""

    bl_label = "Advanced Settings"
    bl_idname = "OBJECT_PT_advanced_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}  # noqa: RUF012
    bl_order = 1

    def draw(self, context: bpy.types.Context) -> None:
        """Draw the panel for the advanced options.

        Args:
            context (bpy.context): Blender Context.

        """
        layout = self.layout

        box = layout.box()

        # dropdown menu for the sd model (sd15 or sdxl so far)
        box.prop(context.scene, "sd_version", text="Stable Diffusion Version:")

        # custom SD checkpoints
        box.prop(context.scene, "checkpoint_path", text="Checkpoint")

        # custom SD resolution
        box.prop(context.scene, "custom_sd_resolution", text="Custom SD Resolution")

        # warning for the user to not go too high with the resolution,
        # especially for the parallel operation,
        # since the resolution will be multiplied by sqrt(num of cameras)
        if (
            int(context.scene.custom_sd_resolution)
            % math.sqrt(int(context.scene.num_cameras))
            != 0
        ):
            box.label(
                text="Warning: Resolution needs to be divisible by sqrt(num_cameras).",
                icon="ERROR",
            )

        if context.scene.sd_version == "sdxl":
            self.panel_sdxl_controlnets(context=context, controlnet_panel=box)

        else:
            self.panel_sd15_controlnets(context=context, controlnet_panel=box)

        """
        # TODO AttributeError: type object 'Panel' has no attribute 'layout'
        """

    def panel_sd15_controlnets(
        self,
        context: bpy.types.Context,
        controlnet_panel: bpy.types.Panel.layout,
    ) -> None:
        """Draw the panel for SD 1.5 ControlNet Paths.

        Args:
            context (bpy.types.Context): Blender Context
            controlnet_panel (bpy.types.Panel.layout): Panel
        """
        # Add advanced settings
        controlnet_panel.label(text="ControlNet Checkpoints:")
        controlnet_panel.prop(context.scene, "depth_controlnet_path", text="Depth Path")

        # disable the canny_controlnet_path option
        # if the mesh complexity is set to LOW
        row = controlnet_panel.row()
        row.enabled = context.scene.mesh_complexity != "LOW"
        row.prop(context.scene, "canny_controlnet_path", text="Canny Path")

        # disable the normal_controlnet_path option
        # if the mesh complexity is set to LOW or MID
        row = controlnet_panel.row()
        row.enabled = context.scene.mesh_complexity == "HIGH"
        row.prop(context.scene, "normal_controlnet_path", text="Normal Path")

        controlnet_panel.label(text="ControlNet Strengths:")
        controlnet_panel.prop(
            context.scene,
            "depth_controlnet_strength",
            text="Depth Strength",
        )

        # disable the canny_controlnet_strength option
        # if the mesh complexity is set to LOW
        row = controlnet_panel.row()
        row.enabled = context.scene.mesh_complexity != "LOW"
        row.prop(context.scene, "canny_controlnet_strength", text="Canny Strength")

        # disable the normal_controlnet_strength option
        # if the mesh complexity is set to LOW or MID
        row = controlnet_panel.row()
        row.enabled = context.scene.mesh_complexity == "HIGH"
        row.prop(
            context.scene,
            "normal_controlnet_strength",
            text="Normal Strength",
        )

    def panel_sdxl_controlnets(
        self,
        context: bpy.types.Context,
        controlnet_panel: bpy.types.Panel.layout,
    ) -> None:
        """Draw the panel for SD XL ControlNet Paths.

        Args:
            context (bpy.types.Context): Blender Context
            controlnet_panel (bpy.types.Panel.layout): Panel
        """
        controlnet_panel.label(text="ControlNet Mode:")

        controlnet_panel.label(text="ControlNet Union Inputs:")
        controlnet_panel.prop(
            context.scene,
            "controlnet_union_path",
            text="ControlNet Union Path",
        )

        controlnet_panel.label(text="ControlNet Strength:")
        controlnet_panel.prop(
            context.scene,
            "union_controlnet_strength",
            text="Union Control Strength",
        )
