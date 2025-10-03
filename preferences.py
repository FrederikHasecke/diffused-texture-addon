from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import bpy

from .mock_context import MockScene

# -----------------
# Very small helpers
# -----------------


def _ensure_pip() -> None:
    try:
        import pip  # noqa: F401
    except Exception:  # noqa: BLE001
        import ensurepip

        ensurepip.bootstrap()
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=False,
    )


def _deps_target_dir() -> Path:
    base = Path(bpy.utils.user_resource("SCRIPTS", path="", create=True))
    target = base / "modules" / "diffusedtexture_deps"
    target.mkdir(parents=True, exist_ok=True)
    return target


# -----------------
# Install models (kept)
# -----------------


class InstallModelsOperator(bpy.types.Operator):  # noqa: D101
    bl_idname = "diffused_texture_addon.install_models"
    bl_label = "Install Models"
    bl_description = "Install the necessary models for DiffusedTexture"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:  # noqa: ARG002, D102
        prefs = bpy.context.preferences.addons[__package__].preferences
        hf_cache_path = prefs.hf_cache_path

        if hf_cache_path:
            os.environ["HF_HOME"] = hf_cache_path

        if not bpy.app.online_access:
            os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            from .diffusedtexture.pipeline.pipeline_builder import (
                create_diffusion_pipeline,
            )  # noqa: PLC0415, RUF100
        except ModuleNotFoundError:
            self.report(
                {"ERROR"},
                "Python dependencies missing. Install Python Dependencies first.",
            )
            return {"CANCELLED"}

        try:
            Path(hf_cache_path).mkdir(parents=True, exist_ok=True)
            mock_scene = MockScene()
            pipe = create_diffusion_pipeline(mock_scene)
            if pipe is not None:
                del pipe
                self.report({"INFO"}, f"Models installed in {hf_cache_path}.")
            else:
                self.report({"ERROR"}, "Failed to create diffusion pipeline.")
                return {"CANCELLED"}
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed: {e!s}")
            return {"CANCELLED"}

        return {"FINISHED"}


# -----------------
# Install Python deps (minimal, pinned to CUDA 12.8)
# -----------------


class InstallDepsOperator(bpy.types.Operator):  # noqa: D101
    bl_idname = "diffused_texture_addon.install_deps"
    bl_label = "Install Python Dependencies"
    bl_description = "Download & install required Python packages (PyTorch CUDA 12.8)"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:  # noqa: ARG002, D102
        if not bpy.app.online_access:
            self.report(
                {"ERROR"},
                "Online access disabled (Preferences > System > Network).",
            )
            return {"CANCELLED"}

        _ensure_pip()
        target = _deps_target_dir()

        # Optional: clean target to avoid leftovers during debugging
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)

        # Make sure our target is importable in this session
        if str(target) not in sys.path:
            sys.path.insert(0, str(target))
        importlib.invalidate_caches()

        # Install everything in ONE resolver pass, pulling torch from CUDA 12.8 channel
        extra_index = "https://download.pytorch.org/whl/cu128"
        pkgs = [
            "torch",
            "scipy",
            "pillow",
            "opencv-python-headless==4.8.1.78",
            "diffusers",
            "transformers",
            "peft",
        ]

        args = [
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
            "--extra-index-url",
            extra_index,
            *pkgs,
        ]
        proc = subprocess.run(  # noqa: S603
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        importlib.invalidate_caches()

        if proc.returncode != 0:
            self.report(
                {"ERROR"},
                "Dependency install failed. Check the console output.",
            )
            return {"CANCELLED"}

        # Minimal import sanity
        try:
            import cv2  # noqa: F401
            import diffusers  # noqa: F401
            import PIL  # noqa: F401
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Installed, but imports failing: {e}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Dependencies installed to {target} (PyTorch CUDA 12.8).",
        )
        return {"FINISHED"}


# -----------------
# Preferences UI (trimmed)
# -----------------


class DiffuseTexPreferences(bpy.types.AddonPreferences):  # noqa: D101
    bl_idname = __package__

    hf_cache_path: bpy.props.StringProperty(
        name="HuggingFace Cache Path",
        description="Custom HuggingFace cache location",
        subtype="DIR_PATH",
        default="",
    )  # type: ignore  # noqa: PGH003

    def draw(self, context: bpy.types.Context) -> None:  # noqa: ARG002, D102
        layout = self.layout

        if not bpy.app.online_access:
            box = layout.box()
            box.label(text="Network:")
            row = box.row()
            row = box.row()
            row.label(text="Online access disabled.", icon="ERROR")
            row.label(text="Enable online access in Preferences > System > Network.")

        layout.prop(self, "hf_cache_path", text="HuggingFace Cache Path")

        deps = layout.box()
        deps.label(text="Python Dependencies")
        r = deps.row()
        r.enabled = bpy.app.online_access
        r.operator(
            InstallDepsOperator.bl_idname,
            text="Install Python Dependencies",
            icon="IMPORT",
        )

        mdl = layout.box()
        mdl.label(text="Models")
        row = mdl.row()
        row.enabled = bpy.app.online_access
        row.operator(
            InstallModelsOperator.bl_idname,
            text="Install Models",
            icon="IMPORT",
        )


# -----------------
# Registration
# -----------------

classes = (InstallModelsOperator, InstallDepsOperator, DiffuseTexPreferences)


def register() -> None:
    for c in classes:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
