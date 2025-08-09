"""Handles the operators of the addon."""

import threading
import time
from pathlib import Path

import bpy
from PIL import Image

from .blender_operations import (
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

    def execute(
        self: "OBJECT_OT_GenerateTexture",
        context: bpy.types.Context,
    ) -> set[str]:
        """Execute the Generation Process.

        Args:
            context (bpy.context): _description_

        Returns:
            set[str]: _description_

        """
        self._done = False
        self._error = None
        self._progress = 0
        self._start_time = time.time()
        self._return_texture = []
        self._output_file = None

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

            def mark_done(success: bool = True, error: str | None = None) -> None:  # noqa: ARG001, FBT001, FBT002
                self._done = True
                if error:
                    self._error = error

            # Progress callback for thread
            def progress_callback(val: int) -> None:
                self._progress = 15 + int(0.85 * val)  # val: 0-100, map to 15-100

            # Start the texture generation in a background thread
            self._thread = threading.Thread(
                target=run_texture_generation,
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
            self.report({"ERROR"}, f"Execution error: {e}")
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        """Update or end Timer.

        Args:
            context (bpy.context): _description_
            event (bpy.types.Event): _description_

        Returns:
            set[str]: _description_
        """
        wm = context.window_manager
        if event.type == "TIMER":
            # Update progress bar from thread progress
            wm.progress_update(self._progress)

            if self._done:
                wm.event_timer_remove(self._timer)
                wm.progress_end()
                context.window.cursor_set("DEFAULT")

                if self._error:
                    self.report({"ERROR"}, f"Texture generation failed: {self._error}")
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

                # TODO: Delete output render folders

                self.report({"INFO"}, "Texture saved successfully.")
                return {"FINISHED"}

        return {"PASS_THROUGH"}
