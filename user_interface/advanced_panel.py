import math

import bpy

from ..controlnet_inputs import (
    get_active_controlnet_inputs,
    is_image_operation_mode,
)
from ..model_support import unsupported_model_message


class OBJECT_PT_AdvancedPanel(bpy.types.Panel):
    """Advanced Settings Panel."""

    bl_label = "Advanced Settings"
    bl_idname = "OBJECT_PT_advanced_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DiffusedTexture"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context: bpy.types.Context) -> None:
        """Draw the panel for the advanced options.

        Args:
            context (bpy.context): Blender Context.

        """
        layout = self.layout

        box = layout.box()

        # dropdown menu for the sd model version
        box.prop(context.scene, "sd_version", text="Model Type:")

        # custom SD checkpoints
        box.prop(context.scene, "checkpoint_path", text="Checkpoint")

        # custom SD resolution
        box.prop(context.scene, "custom_sd_resolution", text="Custom SD Resolution")

        # warning for the user to not go too high with the resolution,
        # especially for the parallel operation,
        # since the resolution will be multiplied by sqrt(num of cameras)
        if (
            is_image_operation_mode(context.scene.operation_mode)
            and int(context.scene.num_cameras) > 0
            and (
                int(context.scene.custom_sd_resolution)
                % math.sqrt(int(context.scene.num_cameras))
                != 0
            )
        ):
            box.label(
                text="Warning: Resolution needs to be divisible by sqrt(num_cameras).",
                icon="ERROR",
            )

        if context.scene.sd_version == "sdxl":
            self.panel_sdxl_controlnets(context=context, controlnet_panel=box)
        elif context.scene.sd_version == "sd15":
            self.panel_sd15_controlnets(context=context, controlnet_panel=box)
        else:
            box.label(
                text="Unsupported model selected.",
                icon="ERROR",
            )
            box.label(text=unsupported_model_message(context.scene.sd_version))
            return

    def panel_sd15_controlnets(
        self,
        context: bpy.types.Context,
        controlnet_panel: bpy.types.Panel,
    ) -> None:
        """Draw the panel for SD 1.5 ControlNet Paths.

        Args:
            context (bpy.types.Context): Blender Context
            controlnet_panel (bpy.types.Panel): Panel
        """
        active_inputs = set(
            get_active_controlnet_inputs(
                context.scene.operation_mode,
                context.scene.mesh_complexity,
            ),
        )

        # Add advanced settings
        controlnet_panel.label(text="ControlNet Checkpoints:")
        row = controlnet_panel.row()
        row.enabled = "depth" in active_inputs
        row.prop(context.scene, "depth_controlnet_path", text="Depth Path")

        row = controlnet_panel.row()
        row.enabled = "canny" in active_inputs
        row.prop(context.scene, "canny_controlnet_path", text="Canny Path")

        row = controlnet_panel.row()
        row.enabled = "normal" in active_inputs
        row.prop(context.scene, "normal_controlnet_path", text="Normal Path")

        controlnet_panel.label(text="ControlNet Strengths:")
        row = controlnet_panel.row()
        row.enabled = "depth" in active_inputs
        row.prop(
            context.scene,
            "depth_controlnet_strength",
            text="Depth Strength",
        )

        row = controlnet_panel.row()
        row.enabled = "canny" in active_inputs
        row.prop(context.scene, "canny_controlnet_strength", text="Canny Strength")

        row = controlnet_panel.row()
        row.enabled = "normal" in active_inputs
        row.prop(
            context.scene,
            "normal_controlnet_strength",
            text="Normal Strength",
        )

    def panel_sdxl_controlnets(
        self,
        context: bpy.types.Context,
        controlnet_panel: bpy.types.Panel,
    ) -> None:
        """Draw the panel for SD XL ControlNet Paths.

        Args:
            context (bpy.types.Context): Blender Context
            controlnet_panel (bpy.types.Panel): Panel
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
