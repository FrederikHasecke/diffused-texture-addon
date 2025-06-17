import os
from pathlib import Path

import bpy

from .mock_context import MockUpContext


class InstallModelsOperator(bpy.types.Operator):
    """Operator to install the default models from the add-on menu.

    Args:
        bpy (_type_): _description_

    Returns:
        _type_: _description_
    """

    bl_idname = "diffused_texture_addon.install_models"
    bl_label = "Install Models"
    bl_description = "Install the necessary models for DiffusedTexture"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self) -> set[str]:
        """Execute the default model download.

        Returns:
            set[str]: _description_
        """
        prefs = bpy.context.preferences.addons[__package__].preferences
        hf_cache_path = prefs.hf_cache_path

        if hf_cache_path:
            os.environ["HF_HOME"] = hf_cache_path

        if not bpy.app.online_access:
            os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            from .diffusedtexture.pipeline.pipeline_builder import (
                create_diffusion_pipeline,
            )

            Path(hf_cache_path).mkdir(parents=True)
            pipe = create_diffusion_pipeline(MockUpContext())
            del pipe

            self.report({"INFO"}, f"Models installed in {hf_cache_path}.")
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed: {e!s}")
            return {"CANCELLED"}

        return {"FINISHED"}


class DiffuseTexPreferences(bpy.types.AddonPreferences):
    """Preferences if the Addon in the install window.

    Args:
        bpy (_type_): _description_
    """

    bl_idname = __package__

    hf_cache_path: bpy.props.StringProperty(
        name="HuggingFace Cache Path",
        description="Custom HuggingFace cache location",
        subtype="DIR_PATH",
        default="",
    )  # type: ignore  # noqa: PGH003

    def draw(self) -> None:
        """Draw the preferences menu."""
        layout = self.layout

        box = layout.box()
        box.label(text="Enable online access in Preferences > System > Network.")

        layout.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

        row = layout.row()
        row.enabled = bpy.app.online_access
        row.operator(
            InstallModelsOperator.bl_idname,
            text="Install Models",
            icon="IMPORT",
        )

        if not bpy.app.online_access:
            layout.label(text="Online access disabled.", icon="ERROR")
