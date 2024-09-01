import os
import bpy
import copy
import traceback
from pathlib import Path

from utils import update_uv_maps, get_mesh_objects, update_image_list
from object_ops import move_object_to_origin, calculate_mesh_midpoint
from scene_backup import SceneBackup, clean_scene
from texturegen import first_pass, second_pass, uv_pass


class OBJECT_OT_GenerateTexture(bpy.types.Operator):
    bl_idname = "object.generate_texture"
    bl_label = "Generate Texture"

    def execute(self, context):
        scene = context.scene
        output_path = Path(scene.output_path)

        # Check if an input texture is selected
        input_texture_path = Path(scene.input_texture_path)
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
            input_texture_path = output_path / "first_pass.png"
        else:
            pass  # we are all good

        selected_mesh_name = scene.my_mesh_object
        selected_object = bpy.data.objects.get(selected_mesh_name)

        # Ensure the output path exists
        if not output_path:
            self.report({"ERROR"}, "Output path is not set.")
            return {"CANCELLED"}

        if not output_path.exists():
            self.report({"ERROR"}, "Output path does not exist.")
            return {"CANCELLED"}

        if not any([scene.first_pass, scene.second_pass, scene.refinement_uv_space]):
            self.report({"ERROR"}, "Select at least one pass.")
            return {"CANCELLED"}

        # Save a backup of the current .blend file
        backup_file = output_path / "scene_backup.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(backup_file))

        try:

            # Start progress indicator using context.window_manager
            wm = context.window_manager
            wm.progress_begin(0, 100)

            # Clean the scene, removing all other objects
            clean_scene(scene)

            # Move object to world origin and calculate midpoint
            max_size = calculate_mesh_midpoint(selected_object)
            move_object_to_origin(selected_object)

            texture_final = None

            # Execute texture passes based on user selection
            if scene.first_pass:
                texture_first_pass = first_pass.first_pass(scene, max_size)
                texture_final = copy.deepcopy(texture_first_pass)

                # Save texture as texture_first_pass
                self.save_texture(texture_final, str(output_path / "first_pass.png"))

            if scene.second_pass:

                texture_input = self.load_texture(str(input_texture_path))

                texture_second_pass = second_pass.second_pass(
                    scene, max_size, texture_input
                )
                texture_final = copy.deepcopy(texture_second_pass)

                # Save texture_final as texture_second_pass
                self.save_texture(texture_final, str(output_path / "second_pass.png"))

            if scene.refinement_uv_space:

                if scene.second_pass:
                    texture_input = texture_final
                else:
                    texture_input = self.load_texture(input_texture_path)

                texture_uv_pass = uv_pass.uv_pass(scene, texture_input)
                texture_final = copy.deepcopy(texture_uv_pass)

                # Save texture_final as texture_uv_pass
                self.save_texture(texture_final, str(output_path / "uv_pass.png"))

            # Process complete
            wm.progress_end()

        except Exception as e:
            # Capture and format the stack trace
            error_message = "".join(
                traceback.format_exception(None, e, e.__traceback__)
            )

            # Report the error to the user in Blender's interface
            self.report(
                {"ERROR"}, f"An error occurred: {str(e)}\nDetails:\n{error_message}"
            )

            # End the progress indicator
            wm.progress_end()

            # Optionally, print the error message to the console for detailed inspection
            print(error_message)

            return {"CANCELLED"}

        finally:
            # Restore the original scene by reloading the backup file
            bpy.ops.wm.open_mainfile(filepath=backup_file)

            # Assign the texture_final to the object
            if texture_final:
                self.assign_texture_to_object(selected_object, texture_final)

        return {"FINISHED"}

    def save_texture(self, texture, filepath):
        """Save the texture to a file."""
        if texture:
            texture.filepath_raw = filepath
            texture.file_format = "PNG"
            texture.save()

    def load_texture(self, filepath):
        """Load a texture from a file."""
        if os.path.exists(filepath):
            return bpy.data.images.load(filepath)
        else:
            self.report({"ERROR"}, f"Texture file {filepath} not found.")
            return None

    def assign_texture_to_object(self, obj, texture):
        """Assign the final texture to the object's material."""
        if obj and texture:
            if obj.data.materials:
                material = obj.data.materials[0]
            else:
                material = bpy.data.materials.new(name="Material")
                obj.data.materials.append(material)

            if material.node_tree:
                nodes = material.node_tree.nodes
                bsdf = nodes.get("Principled BSDF")

                if bsdf:
                    texture_node = nodes.new(type="ShaderNodeTexImage")
                    texture_node.image = texture
                    material.node_tree.links.new(
                        bsdf.inputs["Base Color"], texture_node.outputs["Color"]
                    )
                else:
                    self.report({"ERROR"}, "Could not find Principled BSDF node.")
            else:
                self.report({"ERROR"}, "Material has no node tree.")


class OBJECT_OT_SelectPipette(bpy.types.Operator):
    bl_idname = "object.select_pipette"
    bl_label = "Select Object with Pipette"

    def execute(self, context):
        context.scene.my_mesh_object = context.object.name
        return {"FINISHED"}
