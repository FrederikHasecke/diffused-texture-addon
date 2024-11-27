bl_info = {
    "name": "DiffusedTexture",
    "description": "Generate Diffuse Textures on Meshes with Stable Diffusion.",
    "author": "Frederik Hasecke",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > DiffusedTexture",
    "category": "Material",
}

import bpy
import os
import sys

# zip it due File path length limits on Windows
package_path = os.path.join(os.path.dirname(__file__), "python_packages.zip")
if package_path not in sys.path:
    sys.path.append(package_path)

from .properties import register_properties, unregister_properties
from .operators import OBJECT_OT_GenerateTexture, OBJECT_OT_SelectPipette
from .panel import (
    OBJECT_PT_MainPanel,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
    OBJECT_PT_LoRAPanel,
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

        # Example logic for model installation (replace with your actual implementation)
        try:

            # Import after setting HF_HOME
            import diffusers
            from .diffusedtexture.diffusers_utils import create_first_pass_pipeline

            # Safely create the cache directory if it does not exist
            if hf_cache_path:
                os.makedirs(hf_cache_path, exist_ok=True)

            # Create the pipeline (replace with your actual model setup logic)
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
    bl_idname = "diffused_texture_addon"

    # Path setting for HuggingFace cache directory
    hf_cache_path: bpy.props.StringProperty(
        name="HuggingFace Cache Path",
        description="Path to a custom HuggingFace cache directory",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout

        # HuggingFace Cache Path setting
        layout.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

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
    OBJECT_PT_LoRAPanel,
    OBJECT_PT_IPAdapterPanel,
    OBJECT_OT_OpenNewIPAdapterImage,
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
