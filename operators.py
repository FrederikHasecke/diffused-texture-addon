"""Handles the operators of the addon."""

import threading
import time

import bpy

from .blender_operations import (
    bake_uv_views,
    bpy_img_to_numpy,
    extract_process_parameters_from_context,
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
        self._start_time = time.time()

        # Start the rendering process in the main thread (blocking)
        try:
            selected_obj_name = context.scene.my_mesh_object
            selected_obj = bpy.data.objects.get(selected_obj_name)

            # Backup the scene and isolate the object
            scene_backup = prepare_scene(selected_obj)

            if context.scene.operation_mode != "UV":
                # Render views and save to folders
                render_img_folders, cameras = render_views(context, selected_obj)
            else:
                render_img_folders, cameras = bake_uv_views(context, selected_obj)

            # Put the process parameters from the blender context into a dataclass
            process_parameter = extract_process_parameters_from_context(context)

            # if a input texture exists, turn it into an NDArray
            if hasattr(context.scene, "input_texture"):
                input_texture = bpy_img_to_numpy(context.scene.input_texture)
            else:
                input_texture = None

            # Restore the scene after rendering
            restore_scene(scene_backup, cameras)

            # Start the texture generation in a background thread
            self._thread = threading.Thread(
                target=run_texture_generation,
                args=(process_parameter, render_img_folders, input_texture),
                daemon=True,
            )
            self._thread.start()

            wm = context.window_manager
            wm.progress_begin(0, 100)
            self._timer = wm.event_timer_add(0.5, window=context.window)
            wm.modal_handler_add(self)

        except Exception as e:  # noqa: BLE001
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
        if event.type == "TIMER" and self._done:
            context.window_manager.event_timer_remove(self._timer)
            context.window_manager.progress_end()

            if self._error:
                self.report({"ERROR"}, f"Texture generation failed: {self._error}")
                return {"CANCELLED"}

            self.report({"INFO"}, "Texture saved successfully.")
            return {"FINISHED"}

        return {"PASS_THROUGH"}
