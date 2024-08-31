import os
import bpy
from utils import update_uv_maps, get_mesh_objects, update_image_list
from object_ops import move_object_to_origin, calculate_mesh_midpoint
from scene_backup import SceneBackup, clean_scene
from texturegen import first_pass, second_pass, uv_pass


class OBJECT_OT_GenerateTexture(bpy.types.Operator):
    bl_idname = "object.generate_texture"
    bl_label = "Generate Texture"

    def execute(self, context):
        scene = context.scene
        output_path = scene.output_path

        # Check if an input texture is selected
        input_texture_path = scene.input_texture_path
        if (
            not input_texture_path
            and (scene.second_pass or scene.refinement_uv_space)
            and not scene.first_pass
        ):
            self.report(
                {"ERROR"},
                "No input texture selected. Please select a texture for img2img or texture2texture pass.",
            )
            return {"CANCELLED"}
        elif (
            not input_texture_path
            and (scene.second_pass or scene.refinement_uv_space)
            and scene.first_pass
        ):
            input_texture_path = output_path + "first_pass.png"
        else:
            pass  # we are all good

        selected_mesh_name = scene.my_mesh_object
        selected_object = bpy.data.objects.get(selected_mesh_name)

        # Ensure the output path exists
        if not output_path:
            self.report({"ERROR"}, "Output path is not set.")
            return {"CANCELLED"}

        if not os.path.exists(output_path):
            self.report({"ERROR"}, "Output path does not exist.")
            return {"CANCELLED"}

        if not any([scene.first_pass, scene.second_pass, scene.refinement_uv_space]):
            self.report({"ERROR"}, "Select at least one pass.")
            return {"CANCELLED"}

        # Save a backup of the current .blend file
        backup_file = os.path.join(output_path, "scene_backup.blend")
        bpy.ops.wm.save_as_mainfile(filepath=backup_file)

        try:
            # Start progress indicator using context.window_manager
            wm = context.window_manager
            wm.progress_begin(0, 100)

            # Clean the scene, removing all other objects
            clean_scene(scene)

            # Move object to world origin and calculate midpoint
            max_size = calculate_mesh_midpoint(selected_object)
            move_object_to_origin(selected_object)

            # Execute texture passes based on user selection
            if scene.first_pass:
                first_pass.first_pass(scene, max_size)
            if scene.second_pass:
                second_pass.second_pass(scene, max_size)
            if scene.refinement_uv_space:
                uv_pass.uv_pass(scene)

            # Process complete
            wm.progress_end()

        except Exception as e:
            self.report({"ERROR"}, f"An error occurred: {e}")
            wm.progress_end()
            return {"CANCELLED"}

        finally:
            # Restore the original scene by reloading the backup file
            # bpy.ops.wm.open_mainfile(filepath=backup_file)
            pass  # We need to see what happend

        return {"FINISHED"}


class OBJECT_OT_SelectPipette(bpy.types.Operator):
    bl_idname = "object.select_pipette"
    bl_label = "Select Object with Pipette"

    def execute(self, context):
        context.scene.my_mesh_object = context.object.name
        return {"FINISHED"}
