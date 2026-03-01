"""Handles the operators of the addon."""

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import bpy
import numpy as np
from numpy.typing import NDArray

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from .blender_operations import (
    ProcessParameter,
    apply_texture,
    bake_uv_views,
    bpy_img_to_numpy,
    extract_process_parameter_from_context,
    prepare_scene,
    render_views,
    restore_scene,
)
from .texture_generation import run_texture_generation


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

    def _run_texture_generation_thread(  # noqa: PLR0913
        self,
        process_parameter: ProcessParameter,
        render_img_folders: dict[str, NDArray | str],
        progress_callback: Callable[[int], None],
        mark_done: Callable[[bool, str | None], None],
        return_texture: list[NDArray[np.uint8]],
        input_texture: NDArray[np.float32] | None = None,
    ) -> None:
        """Thread wrapper that surfaces errors back to modal state."""
        try:
            run_texture_generation(
                process_parameter,
                render_img_folders,
                progress_callback,
                mark_done,
                return_texture,
                input_texture,
            )
        except Exception as exc:  # noqa: BLE001
            mark_done(success=False, error=str(exc))

    def execute(  # noqa: PLR0915
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

        self._done = False
        self._error = None
        self._progress = 0
        self._start_time = time.time()
        self._return_texture = []
        self._output_file = None
        self.render_img_folders = None
        self._set_scene_status(
            context.scene,
            running=True,
            done=False,
            error="",
        )

        # Start progress bar for the whole process
        wm = context.window_manager
        wm.progress_begin(0, 100)
        context.window.cursor_set("WAIT")

        try:
            selected_obj_name = context.scene.my_mesh_object
            selected_obj = bpy.data.objects.get(selected_obj_name)

            # Backup the scene and isolate the object
            scene_backup = prepare_scene(selected_obj)

            # Rendering progress (simulate with steps)
            if context.scene.operation_mode != "UV":
                # Render views and save to folders
                wm.progress_update(5)
                render_img_folders, cameras = render_views(context, selected_obj)
                wm.progress_update(10)
            else:
                wm.progress_update(5)
                render_img_folders, cameras = bake_uv_views(context, selected_obj)
                wm.progress_update(10)
            self.render_img_folders = render_img_folders

            # Restore the scene after rendering
            restore_scene(scene_backup, cameras)

            # Put the process parameter from context to a dataclass for thread safety
            process_parameter = extract_process_parameter_from_context(context)

            self._output_file = process_parameter.output_path

            # if an input texture exists, turn it into an NDArray
            if context.scene.input_texture:
                input_texture = bpy_img_to_numpy(context.scene.input_texture)
            else:
                input_texture = None

            wm.progress_update(15)

            def mark_done(success: bool = True, error: str | None = None) -> None:  # noqa: FBT001, FBT002
                self._done = True
                if not success:
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
                    render_img_folders,
                    progress_callback,
                    mark_done,
                    self._return_texture,
                    input_texture,
                ),
                daemon=True,
            )
            self._thread.start()
            self._timer = wm.event_timer_add(0.5, window=context.window)
            wm.modal_handler_add(self)

        except Exception as e:  # noqa: BLE001
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

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        """Run modal opertations outside of threading."""
        wm = context.window_manager
        if event.type == "TIMER":
            # Update progress bar from thread progress
            wm.progress_update(self._progress)

            if (
                self._thread is not None
                and not self._thread.is_alive()
                and not self._done
            ):
                self._error = "Texture generation thread exited unexpectedly."
                self._done = True

            if self._done:
                wm.event_timer_remove(self._timer)
                wm.progress_end()
                context.window.cursor_set("DEFAULT")

                if self._error:
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

                # Save the resulting texture
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = (
                    Path(self._output_file) / f"output_texture_{timestamp}.png"
                )
                Image.fromarray(self._return_texture[0]).save(output_path)

                # apply the texture to the selected object
                apply_texture(
                    context,
                    output_path,
                )

                self.report({"INFO"}, "Texture saved successfully.")

                # delete the temporary render folders
                for render_type in self.render_img_folders:
                    render_type_folder = self.render_img_folders[render_type]
                    if Path(render_type_folder).is_dir():
                        shutil.rmtree(render_type_folder, ignore_errors=True)

                # delete the parent folders if they are empty
                parent_folder = Path(self.render_img_folders["depth"]).parent
                if (parent_folder.is_dir() and not any(parent_folder.iterdir())) or (
                    parent_folder.is_dir()
                    and list(parent_folder.iterdir())
                    == [
                        parent_folder / "render_.exr",
                    ]
                ):
                    shutil.rmtree(parent_folder, ignore_errors=True)

                self._set_scene_status(
                    context.scene,
                    running=False,
                    done=True,
                    error="",
                )

                return {"FINISHED"}

        return {"PASS_THROUGH"}
