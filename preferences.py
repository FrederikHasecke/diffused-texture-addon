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
        status.label(text="Runtime Capability")
        status.label(
            text=(
                "Selected backend: "
                f"{runtime_capability.selected_torch_choice} "
                f"(installs {runtime_capability.torch_install_channel})"
            ),
        )
        status.label(
            text=(
                "Active deps env: "
                f"{runtime_capability.active_deps_path or 'unavailable'}"
            ),
        )
        if runtime_capability.cycles_ui_status == "gpu":
            status.label(
                text=(
                    "Cycles capability: "
                    f"{runtime_capability.cycles_backend or 'GPU'} "
                    f"({runtime_capability.scene_render_device})"
                ),
            )
        elif runtime_capability.cycles_ui_status == "cpu":
            status.label(text="Cycles capability: CPU")
        else:
            status.label(
                text=(
                    "Cycles capability: inconclusive "
                    "(resolved at render time)"
                ),
            )

        torch_text = runtime_capability.torch_version or "unavailable"
        if runtime_capability.torch_cuda_build:
            torch_text = (
                f"{torch_text}, "
                f"CUDA build {runtime_capability.torch_cuda_build}"
            )
        status.label(text=f"Imported torch: {torch_text}")
        status.label(
            text=(
                "Diffusion runtime: "
                f"{runtime_capability.diffusion_device or 'unavailable'}"
            ),
        )
        if runtime_capability.diffusion_environment_warning:
            status.label(
                text=runtime_capability.diffusion_environment_warning,
                icon="ERROR",
            )
        elif runtime_capability.diffusion_dependencies_importable:
            status.label(text="Diffusion dependencies importable.", icon="CHECKMARK")
        else:
            status.label(
                text=(
                    "Diffusion dependencies missing. "
                    "Install Python Dependencies and restart Blender."
                ),
                icon="ERROR",
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
