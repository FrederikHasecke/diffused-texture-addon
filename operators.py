"""Handles the operators of the addon."""

import threading
import time

import bpy

from .blender_operations import prepare_scene, render_views
from .texture_generation import run_texture_generation
from .utils import restore_scene


class OBJECT_OT_GenerateTexture(bpy.types.Operator):
    """Start texture generation in a background thread"""

    bl_idname = "object.generate_texture"
    bl_label = "Generate Texture"

    _timer = None
    _thread = None
    _done = False
    _error = None
    _output_file = None
    _start_time = None
    _last_progress = 0

    def execute(self, context):
        self._done = False
        self._error = None
        self._start_time = time.time()

        # Start the rendering process in the main thread (blocking)
        try:
            scene = context.scene
            selected_obj_name = scene.my_mesh_object
            selected_obj = bpy.data.objects.get(selected_obj_name)

            # Backup the scene and isolate the object
            scene_backup = prepare_scene(selected_obj)

            # Render views and save to folders
            render_img_folders = render_views(scene, selected_obj)

            # Restore the scene after rendering
            restore_scene(scene_backup)

            # Start the texture generation in a background thread
            self._thread = threading.Thread(
                target=run_texture_generation,
                args=(scene, render_img_folders),
                daemon=True,
            )
            self._thread.start()

            wm = context.window_manager
            wm.progress_begin(0, 100)
            self._timer = wm.event_timer_add(0.5, window=context.window)
            wm.modal_handler_add(self)

            return {"RUNNING_MODAL"}

        except Exception as e:
            self.report({"ERROR"}, f"Execution error: {e}")
            return {"CANCELLED"}

    def modal(self, context, event):
        if event.type == "TIMER":
            if self._done:
                context.window_manager.event_timer_remove(self._timer)
                context.window_manager.progress_end()

                if self._error:
                    self.report({"ERROR"}, f"Texture generation failed: {self._error}")
                    return {"CANCELLED"}

                self.report({"INFO"}, "Texture saved successfully.")
                return {"FINISHED"}

        return {"PASS_THROUGH"}
