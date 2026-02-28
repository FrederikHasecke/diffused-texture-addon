import importlib
import os
import shutil
import sys
from pathlib import Path

import bpy

from ..mock_context import MockScene
from .cuda import normalize_choice, resolve_torch_install, torch_index_url
from .paths import (
    clean_pip_env,
    deps_target_dir,
    ensure_pip,
    make_importable,
    new_deps_target_dir,
    read_pyproject_runtime_deps,
    run,
    set_active_deps_target,
)


class InstallModelsOperator(bpy.types.Operator):
    """Operators to install the Stable Diffusion models."""

    bl_idname = "diffused_texture_addon.install_models"
    bl_label = "Install Models"
    bl_description = "Install the necessary models for DiffusedTexture"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:  # noqa: ARG002
        """Install the necessary models for DiffusedTexture."""
        prefs = bpy.context.preferences.addons[
            ".".join(__package__.split(".")[:-1])
        ].preferences
        hf_cache_path = prefs.hf_cache_path

        if hf_cache_path:
            os.environ["HF_HOME"] = hf_cache_path
        if not bpy.app.online_access:
            os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            from ..diffusedtexture.pipeline.pipeline_builder import (
                create_diffusion_pipeline,
            )
        except ModuleNotFoundError:
            self.report(
                {"ERROR"},
                "Python dependencies missing. Install Python Dependencies first.",
            )
            return {"CANCELLED"}

        try:
            if hf_cache_path:
                Path(hf_cache_path).mkdir(parents=True, exist_ok=True)
            pipe = create_diffusion_pipeline(MockScene())
            if pipe is not None:
                del pipe
                dest = hf_cache_path or "the default HF cache"
                self.report({"INFO"}, f"Models installed in {dest}.")
            else:
                self.report({"ERROR"}, "Failed to create diffusion pipeline.")
                return {"CANCELLED"}
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed: {e!s}")
            return {"CANCELLED"}

        return {"FINISHED"}


class InstallDepsOperator(bpy.types.Operator):
    """Install Operator for the Python dependencies."""

    bl_idname = "diffused_texture_addon.install_deps"
    bl_label = "Install Python Dependencies"
    bl_description = (
        "Download & install required Python packages (CUDA/ROCm/CPU selectable)"
    )
    bl_options = {"REGISTER", "INTERNAL"}

    @staticmethod
    def _sanity_imports(target: Path, env: dict[str, str]) -> tuple[int, str]:
        path = str(target).replace("\\", "\\\\")
        script = (
            f"import site; site.addsitedir(r'{path}');"
            "import accelerate, cv2, diffusers, PIL, safetensors, torch, transformers"
        )
        return run([sys.executable, "-c", script], env=env)

    def execute(self, context: bpy.types.Context) -> set[str]:  # noqa: ARG002
        """Execute the installation of dependencies."""
        if not bpy.app.online_access:
            self.report(
                {"ERROR"},
                "Online access disabled (Preferences > System > Network).",
            )
            return {"CANCELLED"}

        prefs = bpy.context.preferences.addons[
            ".".join(__package__.split(".")[:-1])
        ].preferences
        channel = normalize_choice(prefs.cuda_variant)
        install_channel, torch_requirement, install_note = resolve_torch_install(
            channel,
            platform=sys.platform,
            blender_version=bpy.app.version,
        )
        index_url, label = torch_index_url(
            install_channel,
        )
        ensure_pip()

        previous_target = deps_target_dir()
        target = new_deps_target_dir()

        # 0) Resolve runtime deps directly from pyproject (must NOT include torch/bpy)
        addon_root = Path(__file__).resolve().parents[1]
        pyproject = addon_root / "pyproject.toml"
        runtime_pkgs = read_pyproject_runtime_deps(pyproject)

        env = clean_pip_env()

        # All deps at once
        rc, out = run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--upgrade",
                "--prefer-binary",
                "--no-warn-script-location",
                "--upgrade-strategy",
                "only-if-needed",
                "--target",
                str(target),
                "--index-url",
                "https://pypi.org/simple",
                "--extra-index-url",
                index_url,
                "--only-binary",
                ":all:",
                "--trusted-host",
                "pypi.org",
                "--trusted-host",
                "files.pythonhosted.org",
                "--trusted-host",
                "download.pytorch.org",
                torch_requirement,
                *runtime_pkgs,
            ],
            env=env,
        )

        if rc != 0:
            shutil.rmtree(target, ignore_errors=True)
            self.report({"ERROR"}, f"Dependency install failed.\n{out}")
            return {"CANCELLED"}

        rc, out = self._sanity_imports(target, env)
        if rc != 0:
            shutil.rmtree(target, ignore_errors=True)
            self.report({"ERROR"}, f"Installed, but imports failing:\n{out}")
            return {"CANCELLED"}

        try:
            set_active_deps_target(target)
        except Exception as e:  # noqa: BLE001
            self.report(
                {"ERROR"}, f"Installed, but failed to activate dependencies: {e}"
            )
            return {"CANCELLED"}
        make_importable(target)
        importlib.invalidate_caches()

        msg = f"Dependencies installed to {target} ({label})."
        if previous_target != target:
            msg = f"{msg} Active environment updated."
        if install_note:
            msg = f"{msg} {install_note}"
        self.report({"INFO"}, msg)
        return {"FINISHED"}
