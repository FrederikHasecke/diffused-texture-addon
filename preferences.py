import bpy

from .installer.cuda import CUDA_ENUM_ITEMS
from .installer.operators import InstallDepsOperator, InstallModelsOperator


class DiffuseTexPreferences(bpy.types.AddonPreferences):
    """Preferences for the DiffusedTexture addon."""

    bl_idname = __package__

    hf_cache_path: bpy.props.StringProperty(
        name="HuggingFace Cache Path",
        description="Custom HuggingFace cache location",
        subtype="DIR_PATH",
        default="",
    )  # type: ignore  # noqa: PGH003

    cuda_variant: bpy.props.EnumProperty(  # type: ignore  # noqa: PGH003
        name="PyTorch build",
        description="Choose CUDA/ROCm/CPU for PyTorch (or auto-detect)",
        items=CUDA_ENUM_ITEMS,
        default="AUTO",
    )

    def draw(self, context: bpy.types.Context) -> None:  # noqa: ARG002
        """Draw the preferences UI."""
        layout = self.layout

        if not bpy.app.online_access:
            box = layout.box()
            row = box.row()
            row.label(text="Online access disabled.", icon="ERROR")
            row = box.row()
            row.label(text="Enable in Preferences > System > Network.")

        deps = layout.box()
        deps.label(text="Python Dependencies")
        deps.prop(self, "cuda_variant", text="PyTorch build")
        r = deps.row()
        r.enabled = bpy.app.online_access
        r.operator(
            InstallDepsOperator.bl_idname,
            text="Install Python Dependencies (Requires Restart of Blender)",
            icon="IMPORT",
        )

        mdl = layout.box()
        mdl.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

        r = mdl.row()
        r.enabled = bpy.app.online_access
        r.operator(
            InstallModelsOperator.bl_idname,
            text="Install Basic Models",
            icon="IMPORT",
        )


classes = (DiffuseTexPreferences, InstallDepsOperator, InstallModelsOperator)


def register() -> None:
    for c in classes:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
