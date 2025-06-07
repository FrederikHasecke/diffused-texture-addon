import bpy
import os

from .mock_scene import MockUpScene


class InstallModelsOperator(bpy.types.Operator):
    bl_idname = "diffused_texture_addon.install_models"
    bl_label = "Install Models"
    bl_description = "Install the necessary models for DiffusedTexture"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        prefs = bpy.context.preferences.addons[__package__].preferences
        hf_cache_path = prefs.hf_cache_path

        if hf_cache_path:
            os.environ["HF_HOME"] = hf_cache_path

        if not bpy.app.online_access:
            os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            import diffusers  # noqa: F401 Justification, diffusers needs to be imported for home set
            from .diffusedtexture.pipeline.pipeline_builder import (
                create_diffusion_pipeline,
            )

            os.makedirs(hf_cache_path, exist_ok=True)
            pipe = create_diffusion_pipeline(MockUpScene())
            del pipe

            self.report({"INFO"}, f"Models installed in {hf_cache_path}.")
        except Exception as e:
            self.report({"ERROR"}, f"Failed: {str(e)}")
            return {"CANCELLED"}

        return {"FINISHED"}


class DiffuseTexPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    hf_cache_path: bpy.props.StringProperty(
        name="HuggingFace Cache Path",
        description="Custom HuggingFace cache location",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Enable online access in Preferences > System > Network.")

        layout.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

        row = layout.row()
        row.enabled = bpy.app.online_access
        row.operator(
            InstallModelsOperator.bl_idname, text="Install Models", icon="IMPORT"
        )

        if not bpy.app.online_access:
            layout.label(text="Online access disabled.", icon="ERROR")
