import bpy

from .installer.cuda import CUDA_ENUM_ITEMS
from .installer.operators import InstallDepsOperator, InstallModelsOperator
from .model_support import DEFAULT_SD_VERSION, get_sd_version_enum_items
from .runtime_capability import get_runtime_capability


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
        name="Dependency backend",
        description=(
            "Choose the PyTorch dependency backend for diffusion "
            "(or auto-detect)"
        ),
        items=CUDA_ENUM_ITEMS,
        default="AUTO",
    )

    install_model_sd_version: bpy.props.EnumProperty(  # type: ignore  # noqa: PGH003
        name="Model family",
        description="Choose which supported model family to provision",
        items=get_sd_version_enum_items(),
        default=DEFAULT_SD_VERSION,
    )

    def draw(self, context: bpy.types.Context) -> None:
        """Draw the preferences UI."""
        layout = self.layout
        runtime_capability = get_runtime_capability(
            context,
            torch_choice=self.cuda_variant,
        )

        if not bpy.app.online_access:
            box = layout.box()
            row = box.row()
            row.label(text="Online access disabled.", icon="ERROR")
            row = box.row()
            row.label(text="Enable in Preferences > System > Network.")

        deps = layout.box()
        deps.label(text="Python Dependencies")
        deps.prop(self, "cuda_variant", text="Dependency backend")
        r = deps.row()
        r.enabled = bpy.app.online_access
        r.operator(
            InstallDepsOperator.bl_idname,
            text="Install Python Dependencies (Requires Restart of Blender)",
            icon="IMPORT",
        )
        status = deps.box()
        status.label(text="Current Runtime Capability")
        status.label(
            text=f"Dependency backend: {runtime_capability.torch_install_channel}",
        )
        if runtime_capability.scene_render_device == "CPU":
            status.label(text="Cycles render: CPU")
        elif runtime_capability.cycles_backend is not None:
            status.label(
                text=(
                    "Cycles render: "
                    f"{runtime_capability.cycles_backend} "
                    f"({runtime_capability.scene_render_device})"
                ),
            )
        else:
            status.label(text="Cycles render: unavailable")
        status.label(
            text=(
                "Diffusion device: "
                f"{runtime_capability.diffusion_device or 'unavailable'}"
            ),
        )
        status.label(
            text=(
                "Generation ready."
                if runtime_capability.can_generate
                else runtime_capability.message
            ),
            icon="CHECKMARK" if runtime_capability.can_generate else "ERROR",
        )

        mdl = layout.box()
        mdl.label(text="Models")
        mdl.prop(self, "install_model_sd_version", text="Model family")
        mdl.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

        r = mdl.row()
        r.enabled = bpy.app.online_access
        r.operator(
            InstallModelsOperator.bl_idname,
            text="Install Selected Basic Model",
            icon="IMPORT",
        )


classes = (DiffuseTexPreferences, InstallDepsOperator, InstallModelsOperator)


def register() -> None:
    for c in classes:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
