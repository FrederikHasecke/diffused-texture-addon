"""Handles the operators of the addon."""

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import bpy
import numpy as np
from numpy.typing import NDArray

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from .blender_operations import (
    ProcessParameter,
    UVPassAssets,
    apply_texture,
    bpy_img_to_numpy,
    build_uv_pass_assets,
    extract_process_parameter_from_context,
    prepare_scene,
    render_views,
    restore_scene,
)
from .diagnostics import get_logger
from .model_support import require_supported_sd_version
from .runtime_capability import get_runtime_capability
from .texture_generation import load_multiview_images, run_texture_generation

_logger = get_logger("operators")
CANCELLED_BY_USER_MESSAGE = "Cancelled by user."


def _raise_selected_object_not_found(selected_obj_name: str) -> None:
    msg = f"Selected object '{selected_obj_name}' was not found."
    raise ValueError(msg)


def _is_texture_generation_cancelled_error(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "TextureGenerationCancelledError"


class OBJECT_OT_GenerateTexture(bpy.types.Operator):
    """Start texture generation in a background thread."""

    bl_idname = "object.generate_texture"
    bl_label = "Generate Texture"

    _timer = None
    _thread = None
    _done = False
    _error = None
    _output_file = None
    _start_time = None
    _last_progress = 0
    _progress = 0  # 0-100
    _run_id = ""
    _cancelled = False

    def _finalize_generation(  # noqa: C901
        self,
        context: bpy.types.Context,
    ) -> set[str]:
        if self._cancelled:
            _logger.info("Texture generation cancelled. run_id=%s", self._run_id)
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error=CANCELLED_BY_USER_MESSAGE,
            )
            self.report({"INFO"}, "Texture generation cancelled.")
            return {"CANCELLED"}

        if self._error:
            _logger.error(
                "Texture generation failed. run_id=%s error=%s",
                self._run_id,
                self._error,
            )
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error=self._error,
            )
            self.report({"ERROR"}, f"Texture generation failed: {self._error}")
            return {"CANCELLED"}

        if not self._return_texture:
            msg = "Texture generation completed without a texture result."
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error=msg,
            )
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if Image is None:
            msg = "Pillow is not available."
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error=msg,
            )
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if self._output_file is None:
            msg = "Output path is not set."
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error=msg,
            )
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(self._output_file) / f"output_texture_{timestamp}.png"
        Image.fromarray(self._return_texture[0]).save(output_path)

        apply_texture(
            context,
            str(output_path),
        )

        duration = time.time() - self._start_time if self._start_time else 0.0
        _logger.info(
            (
                "Texture generation completed successfully. run_id=%s output=%s "
                "duration=%.2fs"
            ),
            self._run_id,
            output_path,
            duration,
        )
        self.report({"INFO"}, "Texture saved successfully.")

        if self.render_img_folders is not None:
            for render_type_folder in self.render_img_folders.values():
                if (
                    isinstance(render_type_folder, str)
                    and Path(render_type_folder).is_dir()
                ):
                    shutil.rmtree(render_type_folder, ignore_errors=True)

            depth_folder = self.render_img_folders.get("depth")
            if isinstance(depth_folder, str):
                parent_folder = Path(depth_folder).parent
                if (parent_folder.is_dir() and not any(parent_folder.iterdir())) or (
                    parent_folder.is_dir()
                    and list(parent_folder.iterdir()) == [parent_folder / "render_.exr"]
                ):
                    shutil.rmtree(parent_folder, ignore_errors=True)

        self._set_scene_status(
            context.scene,
            running=False,
            done=True,
            error="",
        )

        return {"FINISHED"}

    def _set_scene_status(
        self,
        scene: bpy.types.Scene,
        *,
        running: bool,
        done: bool,
        error: str = "",
    ) -> None:
        """Update operator progress properties when available."""
        if hasattr(scene, "diffused_texture_operator_running"):
            scene.diffused_texture_operator_running = running
        if hasattr(scene, "diffused_texture_operator_done"):
            scene.diffused_texture_operator_done = done
        if hasattr(scene, "diffused_texture_operator_error"):
            scene.diffused_texture_operator_error = error

    def _should_cancel(self, context: bpy.types.Context) -> bool:
        return bool(
            getattr(context.scene, "diffused_texture_operator_cancel_requested", False),
        )

    def _run_texture_generation_thread(  # noqa: PLR0913
        self,
        process_parameter: ProcessParameter,
        progress_callback: Callable[[int], None],
        should_cancel: Callable[[], bool],
        mark_done: Callable[..., None],
        generation_inputs: dict[str, list[NDArray[Any]]] | UVPassAssets | None = None,
        return_texture: list[NDArray[np.uint8]] | None = None,
        input_texture: NDArray[np.float32] | None = None,
        uv_assets: UVPassAssets | None = None,
        *,
        multiview_images: dict[str, list[NDArray[Any]]] | None = None,
    ) -> None:
        """Thread wrapper that surfaces errors back to modal state."""
        inputs = (
            generation_inputs if generation_inputs is not None else multiview_images
        )
        if inputs is None:
            inputs = {}
        textures = return_texture if return_texture is not None else []
        try:
            run_texture_generation(
                process_parameter,
                inputs,
                progress_callback,
                should_cancel,
                mark_done,
                textures,
                input_texture,
                uv_assets=uv_assets,
            )
        except Exception as exc:
            if _is_texture_generation_cancelled_error(exc):
                mark_done(
                    success=False,
                    error=CANCELLED_BY_USER_MESSAGE,
                    cancelled=True,
                )
                return

            _logger.exception(
                "Texture generation thread failed. run_id=%s mode=%s",
                self._run_id,
                process_parameter.operation_mode,
            )
            mark_done(success=False, error=str(exc))

    def execute(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self: "OBJECT_OT_GenerateTexture",
        context: bpy.types.Context,
    ) -> set[str]:
        """Execute the Generation Process.

        Args:
            context (bpy.context): _description_

        Returns:
            set[str]: _description_

        """
        try:
            from PIL import Image
        except ModuleNotFoundError:
            Image = None  # noqa: N806

        if Image is None:
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error="Python dependencies missing.",
            )
            self.report(
                {"ERROR"},
                "Python dependencies missing. Open Preferences > Add-ons > DiffusedTexture > Install Python Dependencies.",  # noqa: E501
            )
            return {"CANCELLED"}

        if getattr(context.scene, "diffused_texture_operator_running", False):
            self.report({"WARNING"}, "Texture generation is already running.")
            return {"CANCELLED"}

        self._run_id = uuid4().hex[:8]
        self._done = False
        self._error = None
        self._progress = 0
        self._start_time = time.time()
        self._cancelled = False
        self._return_texture = []
        self._output_file = None
        self.render_img_folders = None
        if hasattr(context.scene, "diffused_texture_operator_cancel_requested"):
            context.scene.diffused_texture_operator_cancel_requested = False
        self._set_scene_status(
            context.scene,
            running=True,
            done=False,
            error="",
        )

        # Start progress bar for the whole process
        wm = context.window_manager
        is_background = getattr(getattr(bpy, "app", None), "background", False)
        if not is_background:
            wm.progress_begin(0, 100)
            context.window.cursor_set("WAIT")

        try:
            selected_obj_name = context.scene.my_mesh_object
            selected_obj = bpy.data.objects.get(selected_obj_name)
            if selected_obj is None:
                _raise_selected_object_not_found(selected_obj_name)

            # Snapshot parameters early so unsupported models fail before rendering.
            process_parameter = extract_process_parameter_from_context(context)
            require_supported_sd_version(process_parameter.sd_version)
            runtime_capability = get_runtime_capability(context)
            if not runtime_capability.can_generate:
                self._set_scene_status(
                    context.scene,
                    running=False,
                    done=False,
                    error=runtime_capability.message,
                )
                self.report({"ERROR"}, runtime_capability.message)
                return {"CANCELLED"}
            self._output_file = process_parameter.output_path
            _logger.info(
                (
                    "Starting texture generation. run_id=%s mode=%s object=%s "
                    "output=%s runtime=%s"
                ),
                self._run_id,
                getattr(
                    process_parameter,
                    "operation_mode",
                    getattr(context.scene, "operation_mode", "unknown"),
                ),
                selected_obj_name,
                process_parameter.output_path,
                runtime_capability.message,
            )

            scene_backup = None
            cameras = []
            mode = getattr(
                process_parameter,
                "operation_mode",
                getattr(context.scene, "operation_mode", "PARALLEL_IMG"),
            )
            uv_assets = None

            try:
                # Backup the scene and isolate the object
                scene_backup = prepare_scene(selected_obj)

                if mode == "UV_PASS":
                    wm.progress_update(5)
                    generation_inputs = build_uv_pass_assets(context, selected_obj)
                    wm.progress_update(10)
                else:
                    wm.progress_update(5)
                    render_img_folders, cameras = render_views(context, selected_obj)
                    generation_inputs = load_multiview_images(render_img_folders)
                    uv_assets = build_uv_pass_assets(context, selected_obj)
                    wm.progress_update(10)
                    self.render_img_folders = render_img_folders
            finally:
                if scene_backup is not None:
                    restore_scene(scene_backup, cameras)

            # if an input texture exists, turn it into an NDArray
            if context.scene.input_texture:
                input_texture = bpy_img_to_numpy(context.scene.input_texture)
            else:
                input_texture = None

            wm.progress_update(15)

            if self._should_cancel(context):
                self._cancelled = True
                return self._finalize_generation(context)

            def mark_done(
                *,
                success: bool = True,
                error: str | None = None,
                cancelled: bool = False,
            ) -> None:
                self._done = True
                self._cancelled = cancelled
                if cancelled:
                    self._error = None
                elif not success:
                    self._error = error or "Texture generation failed."
                elif error:
                    self._error = error

            # Progress callback for thread
            def progress_callback(val: int) -> None:
                self._progress = 15 + int(0.85 * val)  # val: 0-100, map to 15-100

            # Start the texture generation in a background thread
            self._thread = threading.Thread(
                target=self._run_texture_generation_thread,
                args=(
                    process_parameter,
                    progress_callback,
                    lambda: self._should_cancel(context),
                    mark_done,
                    generation_inputs,
                    self._return_texture,
                    input_texture,
                    uv_assets,
                ),
                daemon=True,
            )
            self._thread.start()

            if is_background:
                while self._thread.is_alive():
                    if self._should_cancel(context):
                        self._cancelled = True
                    time.sleep(0.1)

                if not self._done:
                    self._error = "Texture generation thread exited unexpectedly."
                    self._done = True
                    _logger.error(
                        "Texture generation thread exited unexpectedly. run_id=%s",
                        self._run_id,
                    )

                return self._finalize_generation(context)

            self._timer = wm.event_timer_add(0.5, window=context.window)
            wm.modal_handler_add(self)

        except Exception as e:
            _logger.exception(
                "Texture generation setup failed. run_id=%s",
                self._run_id,
            )
            if not is_background:
                wm.progress_end()
                context.window.cursor_set("DEFAULT")
            self._set_scene_status(
                context.scene,
                running=False,
                done=False,
                error=str(e),
            )
            self.report({"ERROR"}, f"Execution error: {e}")
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def modal(
        self,
        context: bpy.types.Context,
        event: bpy.types.Event,
    ) -> set[str]:
        """Run modal opertations outside of threading."""
        wm = context.window_manager
        if event.type == "TIMER":
            # Update progress bar from thread progress
            wm.progress_update(self._progress)

            if self._should_cancel(context):
                self._cancelled = True

            if (
                self._thread is not None
                and not self._thread.is_alive()
                and not self._done
            ):
                self._error = "Texture generation thread exited unexpectedly."
                self._done = True
                _logger.error(
                    (
                        "Texture generation thread exited unexpectedly before "
                        "completion. run_id=%s"
                    ),
                    self._run_id,
                )

            if self._done:
                if self._timer is not None:
                    wm.event_timer_remove(self._timer)
                    self._timer = None
                wm.progress_end()
                context.window.cursor_set("DEFAULT")
                return self._finalize_generation(context)

        return {"PASS_THROUGH"}


class OBJECT_OT_CancelTextureGeneration(bpy.types.Operator):
    """Request cancellation of a running texture generation job."""

    bl_idname = "object.cancel_texture_generation"
    bl_label = "Cancel Texture Generation"

    def execute(self, context: bpy.types.Context) -> set[str]:
        """Request cancellation for the current generation job."""
        if not getattr(context.scene, "diffused_texture_operator_running", False):
            self.report({"WARNING"}, "No texture generation is running.")
            return {"CANCELLED"}

        if hasattr(context.scene, "diffused_texture_operator_cancel_requested"):
            context.scene.diffused_texture_operator_cancel_requested = True
        self.report({"INFO"}, "Cancelling texture generation...")
        return {"FINISHED"}
