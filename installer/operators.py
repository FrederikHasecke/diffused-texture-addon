import importlib
import os
import shutil
import sys
from pathlib import Path

import bpy

from ..diagnostics import get_log_file_path, get_logger
from ..mock_context import MockScene
from ..model_support import MODEL_SUPPORT_MATRIX, require_supported_sd_version
from .cuda import normalize_choice
from .paths import (
    clean_pip_env,
    deps_target_dir,
    ensure_pip,
    make_importable,
    new_deps_target_dir,
    run,
    run_stream,
    set_active_deps_target,
)
from .runtime_matrix import (
    resolve_runtime_spec,
    resolve_torch_install,
    torch_index_url,
)

_logger = get_logger("installer.operators")


def _tail_output(output: str, limit: int = 12) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def _build_failure_message(prefix: str, output: str, return_code: int) -> str:
    message = f"{prefix} (exit code {return_code})."
    tail = _tail_output(output)
    if tail:
        message = f"{message}\nLast output:\n{tail}"

    log_path = get_log_file_path()
    if log_path is not None:
        message = f"{message}\nLog: {log_path}"
    return message


class InstallModelsOperator(bpy.types.Operator):
    """Operators to install the Stable Diffusion models."""

    bl_idname = "diffused_texture_addon.install_models"
    bl_label = "Install Models"
    bl_description = "Install the selected basic model stack for DiffusedTexture"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:  # noqa: ARG002
        """Install the necessary models for DiffusedTexture."""
        prefs = bpy.context.preferences.addons[
            ".".join(__package__.split(".")[:-1])
        ].preferences
        hf_cache_path = prefs.hf_cache_path
        model_version = require_supported_sd_version(prefs.install_model_sd_version)
        model_label = str(MODEL_SUPPORT_MATRIX[model_version]["label"])

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
            _logger.info(
                "Starting model installation. model=%s hf_cache_path=%s",
                model_version,
                hf_cache_path,
            )
            pipe = create_diffusion_pipeline(MockScene(sd_version=model_version))
            if pipe is not None:
                del pipe
                dest = hf_cache_path or "the default HF cache"
                _logger.info(
                    "Model installation completed. model=%s destination=%s",
                    model_version,
                    dest,
                )
                self.report({"INFO"}, f"{model_label} models installed in {dest}.")
            else:
                _logger.error(
                    (
                        "Model installation failed: diffusion pipeline was not "
                        "created. model=%s"
                    ),
                    model_version,
                )
                self.report({"ERROR"}, "Failed to create diffusion pipeline.")
                return {"CANCELLED"}
        except Exception as e:
            _logger.exception("Model installation failed.")
            self.report({"ERROR"}, f"Failed: {e!s}")
            return {"CANCELLED"}

        return {"FINISHED"}


class InstallDepsOperator(bpy.types.Operator):
    """Install Operator for the Python dependencies."""

    bl_idname = "diffused_texture_addon.install_deps"
    bl_label = "Install Python Dependencies"
    bl_description = (
        "Download & install required diffusion Python packages (dependency backend selectable)"  # noqa: E501
    )
    bl_options = {"REGISTER", "INTERNAL"}

    @staticmethod
    def _sanity_imports(target: Path, env: dict[str, str]) -> tuple[int, str]:
        path = str(target).replace("\\", "\\\\")
        script = (
            f"import site; site.addsitedir(r'{path}');"
            "import accelerate, cv2, diffusers, numpy, peft, PIL, "
            "safetensors, torch, transformers"
        )
        return run(
            [sys.executable, "-c", script],
            env=env,
            label="verify dependency imports",
        )

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

        runtime_spec = resolve_runtime_spec(
            blender_version=bpy.app.version,
            python_version=sys.version_info[:2],
        )
        runtime_pkgs = list(runtime_spec.runtime_requirements)

        env = clean_pip_env()
        _logger.info(
            "Starting dependency install. target=%s channel=%s runtime_packages=%s",
            target,
            install_channel,
            runtime_pkgs,
        )

        # All deps at once
        rc, out = run_stream(
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
            label="install Python dependencies",
        )

        if rc != 0:
            shutil.rmtree(target, ignore_errors=True)
            _logger.error(
                "Dependency install failed. rc=%s target=%s output_tail=%s",
                rc,
                target,
                _tail_output(out),
            )
            self.report(
                {"ERROR"},
                _build_failure_message("Dependency install failed", out, rc),
            )
            return {"CANCELLED"}

        rc, out = self._sanity_imports(target, env)
        if rc != 0:
            shutil.rmtree(target, ignore_errors=True)
            _logger.error(
                "Dependency import verification failed. rc=%s target=%s output_tail=%s",
                rc,
                target,
                _tail_output(out),
            )
            self.report(
                {"ERROR"},
                _build_failure_message(
                    "Installed, but imports are still failing",
                    out,
                    rc,
                ),
            )
            return {"CANCELLED"}

        try:
            set_active_deps_target(target)
        except Exception as e:
            _logger.exception("Dependency install completed but activation failed.")
            self.report(
                {"ERROR"},
                f"Installed, but failed to activate dependencies: {e}",
            )
            return {"CANCELLED"}
        make_importable(target)
        importlib.invalidate_caches()

        msg = (
            f"Dependencies installed to {target} ({label}). Restart Blender before "
            "generating textures. The current session may still use stale imports."
        )
        if previous_target != target:
            msg = f"{msg} Active environment updated."
        if install_note:
            msg = f"{msg} {install_note}"
        _logger.info("Dependency install completed successfully. target=%s", target)
        self.report({"INFO"}, msg)
        return {"FINISHED"}
