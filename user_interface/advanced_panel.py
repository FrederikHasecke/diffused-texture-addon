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

    def draw(self, context: bpy.context) -> None:
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

        # If the user selected SDXL, offer a dropdown menu for ControlNets, to switch between the default (multiple paths) and the Controlnet Union (single path)
        if context.scene.sd_version == "sdxl":
            box.label(text="ControlNet Mode:")
            box.prop(context.scene, "controlnet_type", text="")

            # If ControlNet Union is selected, show tick-boxes instead of paths
            if context.scene.controlnet_type == "UNION":
                box.label(text="ControlNet Union Inputs:")
                box.prop(
                    context.scene, "controlnet_union_path", text="ControlNet Union Path"
                )

                box.label(text="ControlNet Strength:")
                box.prop(
                    context.scene,
                    "union_controlnet_strength",
                    text="Union Control Strength",
                )

            # If Multiple ControlNets are selected, show the paths and strengths
            else:
                box.label(text="ControlNet Checkpoints:")
                box.label(text="Change the `Mesh Complexity` to enable more options.")
                box.prop(context.scene, "depth_controlnet_path", text="Depth Path")

                # Disable the canny_controlnet_path option if the mesh complexity is set to LOW
                row = box.row()
                row.enabled = context.scene.mesh_complexity != "LOW"
                row.prop(context.scene, "canny_controlnet_path", text="Canny Path")

                # Disable the normal_controlnet_path option if the mesh complexity is set to LOW or MID
                row = box.row()
                row.enabled = context.scene.mesh_complexity == "HIGH"
                row.prop(context.scene, "normal_controlnet_path", text="Normal Path")

                box.label(text="ControlNet Strengths:")
                box.prop(
                    context.scene, "depth_controlnet_strength", text="Depth Strength"
                )

                # Disable the canny_controlnet_strength option if the mesh complexity is set to LOW
                row = box.row()
                row.enabled = context.scene.mesh_complexity != "LOW"
                row.prop(
                    context.scene, "canny_controlnet_strength", text="Canny Strength"
                )

                # Disable the normal_controlnet_strength option if the mesh complexity is set to LOW or MID
                row = box.row()
                row.enabled = context.scene.mesh_complexity == "HIGH"
                row.prop(
                    context.scene, "normal_controlnet_strength", text="Normal Strength"
                )

        else:
            # Add advanced settings
            box.label(text="ControlNet Checkpoints:")
            box.prop(context.scene, "depth_controlnet_path", text="Depth Path")

            # disable the canny_controlnet_path option if the mesh complexity is set to LOW
            row = box.row()
            row.enabled = context.scene.mesh_complexity != "LOW"
            row.prop(context.scene, "canny_controlnet_path", text="Canny Path")

            # disable the normal_controlnet_path option if the mesh complexity is set to LOW or MID
            row = box.row()
            row.enabled = context.scene.mesh_complexity == "HIGH"
            row.prop(context.scene, "normal_controlnet_path", text="Normal Path")

            box.label(text="ControlNet Strengths:")
            box.prop(context.scene, "depth_controlnet_strength", text="Depth Strength")

            # disable the canny_controlnet_strength option if the mesh complexity is set to LOW
            row = box.row()
            row.enabled = context.scene.mesh_complexity != "LOW"
            row.prop(context.scene, "canny_controlnet_strength", text="Canny Strength")

            # disable the normal_controlnet_strength option if the mesh complexity is set to LOW or MID
            row = box.row()
            row.enabled = context.scene.mesh_complexity == "HIGH"
            row.prop(
                context.scene,
                "normal_controlnet_strength",
                text="Normal Strength",
            )
