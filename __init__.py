import bpy
import os
import sys

from .properties import register_properties, unregister_properties
from .operators import OBJECT_OT_GenerateTexture, OBJECT_OT_SelectPipette
from .panel import (
    OBJECT_PT_MainPanel,
    OBJECT_OT_OpenNewInputImage,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_PT_LoRAPanel,
    OBJECT_PT_AdvancedPanel,
)


class MockUpScene:
    """
    A mockup class to simulate the scene properties for pipeline testing.
    """

    def __init__(self):
        self.num_loras = 0
        self.use_ipadapter = True
        self.ipadapter_strength = 0.5
        self.mesh_complexity = "HIGH"


class InstallModelsOperator(bpy.types.Operator):
    """Operator to install necessary models"""

    bl_idname = "diffused_texture_addon.install_models"
    bl_label = "Install Models"
    bl_description = "Install the necessary models for DiffusedTexture"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        # Retrieve the HuggingFace cache path from preferences
        prefs = bpy.context.preferences.addons[__package__].preferences
        hf_cache_path = prefs.hf_cache_path

        # Set environment variable if path is provided
        if hf_cache_path:
            os.environ["HF_HOME"] = hf_cache_path

        # If blender is set to offline this will not work, but lets not sneak always on onto the user
        if not bpy.app.online_access:
            os.environ["HF_HUB_OFFLINE"] = "1"

        # Logic for model installation
        try:

            # Import after setting HF_HOME
            import diffusers
            from .diffusedtexture.diffusers_utils import create_first_pass_pipeline

            # Safely create the cache directory if it does not exist
            if hf_cache_path:
                os.makedirs(hf_cache_path, exist_ok=True)

            # Create the pipeline
            mockup_scene = MockUpScene()

            pipe = create_first_pass_pipeline(mockup_scene)
            del pipe  # Clean up to avoid memory issues

            self.report({"INFO"}, f"Models installed successfully in {hf_cache_path}.")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to install models: {str(e)}")
            return {"CANCELLED"}

        return {"FINISHED"}


class DiffuseTexPreferences(bpy.types.AddonPreferences):
    # bl_idname = __package__
    bl_idname = __package__

    # Path setting for HuggingFace cache directory
    hf_cache_path: bpy.props.StringProperty(
        name="HuggingFace Cache Path",
        description="Path to a custom HuggingFace cache directory",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout

        # Add a text block to explain that the user needs to explicitly allow online access
        box = layout.box()
        row = box.row()
        row.label(text="Please ensure that Blender is allowed to access the internet")
        row = box.row()
        row.label(text="in order to install models. Do so in:")
        row = box.row()
        row.label(text="Preferences > System > Network > Allow Online Access.")

        # HuggingFace Cache Path setting
        layout.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

        # make the Install Models button unavailable if the "online access" is disabled
        if not bpy.app.online_access:
            layout.label(
                text="Online access is disabled. Enable it in Preferences > System > Network > Allow Online Access."
            )

            row = layout.row()
            row.enabled = False
            row.operator(
                InstallModelsOperator.bl_idname, text="Install Models", icon="IMPORT"
            )

        else:
            # Button to execute the model installation function
            layout.operator(
                InstallModelsOperator.bl_idname, text="Install Models", icon="IMPORT"
            )


classes = [
    DiffuseTexPreferences,
    InstallModelsOperator,
    OBJECT_OT_GenerateTexture,
    OBJECT_OT_SelectPipette,
    OBJECT_PT_MainPanel,
    OBJECT_OT_OpenNewInputImage,
    OBJECT_PT_AdvancedPanel,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_PT_LoRAPanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_properties()


def unregister():
    unregister_properties()
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
