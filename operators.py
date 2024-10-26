import os
import bpy
import copy
import traceback
import numpy as np
from pathlib import Path

from utils import update_uv_maps, get_mesh_objects, update_image_list
from object_ops import move_object_to_origin, calculate_mesh_midpoint
from scene_backup import SceneBackup, clean_scene, clean_object
from diffusedtexture import first_pass, second_pass, third_pass, uv_pass


class OBJECT_OT_GenerateTexture(bpy.types.Operator):
    bl_idname = "object.generate_texture"
    bl_label = "Generate Texture"

    def execute(self, context):
        scene = context.scene
        output_path = Path(scene.output_path)

        # Check if an input texture is selected
        input_texture_path = Path(bpy.path.abspath(scene.input_texture_path))

        if not input_texture_path and (
            scene.operation_mode
            in [
                "IMAGE2IMAGE_PARALLEL",
                "IMAGE2IMAGE_SEQUENTIAL",
                "TEXTURE2TEXTURE_ENHANCEMENT",
            ]
        ):
            self.report(
                {"ERROR"},
                "No input texture selected. Please select a texture for img2img or texture2texture pass.",
            )
            return {"CANCELLED"}

        selected_mesh_name = scene.my_mesh_object
        selected_object = bpy.data.objects.get(selected_mesh_name)

        # Ensure the output path exists
        if not output_path:
            self.report({"ERROR"}, "Output path is not set.")
            return {"CANCELLED"}

        if not output_path.exists():
            self.report({"ERROR"}, "Output path does not exist.")
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
            clean_object(scene)

            # Move object to world origin and calculate midpoint
            max_size = calculate_mesh_midpoint(selected_object)
            move_object_to_origin(selected_object)

            texture_final = None

            # Execute texture passes based on user selection
            if scene.operation_mode == "TEXT2IMAGE_PARALLEL":
                texture_first_pass = first_pass.first_pass(scene, 1.5 * max_size)

                # flip along the v axis
                texture_first_pass = texture_first_pass[::-1]

                texture_final = copy.deepcopy(texture_first_pass)

                # Save texture as texture_first_pass
                self.save_texture(texture_final, str(output_path / "first_pass.png"))
                self.save_texture(texture_final, str(output_path / "final_texture.png"))

            elif scene.operation_mode == "IMAGE2IMAGE_PARALLEL":

                texture_input = self.load_texture(str(input_texture_path))

                texture_second_pass = second_pass.second_pass(
                    scene, 1.5 * max_size, texture_input
                )

                # flip along the v axis
                texture_second_pass = texture_second_pass[::-1]

                texture_final = copy.deepcopy(texture_second_pass)

                # # flip along the v axis
                # texture_final = texture_final[::-1]

                # Save texture as texture_first_pass
                self.save_texture(texture_final, str(output_path / "second_pass.png"))
                self.save_texture(texture_final, str(output_path / "final_texture.png"))

            elif scene.operation_mode == "IMAGE2IMAGE_SEQUENTIAL":

                texture_input = self.load_texture(str(input_texture_path))

                texture_third_pass = third_pass.third_pass(
                    scene, 1.5 * max_size, texture_input
                )

                texture_final = copy.deepcopy(texture_third_pass)

                # flip along the v axis
                texture_final = texture_final[::-1]

                # Save texture_final as texture_second_pass
                self.save_texture(texture_final, str(output_path / "third_pass.png"))
                self.save_texture(texture_final, str(output_path / "final_texture.png"))

            elif scene.operation_mode == "TEXTURE2TEXTURE_ENHANCEMENT":

                texture_input = self.load_texture(str(input_texture_path))

                texture_uv_pass = uv_pass.uv_pass(scene, texture_input)
                texture_final = copy.deepcopy(texture_uv_pass)

                # Save texture_final as texture_uv_pass
                self.save_texture(texture_final, str(output_path / "uv_pass.png"))
                self.save_texture(texture_final, str(output_path / "final_texture.png"))

            else:
                raise NotImplementedError(
                    "Only the four modes 'TEXT2IMAGE_PARALLEL', 'IMAGE2IMAGE_PARALLEL', 'IMAGE2IMAGE_SEQUENTIAL', 'TEXTURE2TEXTURE_ENHANCEMENT' are implemented"
                )

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

            # TODO: do not restore (DEBUG)

            # # Restore the original scene by reloading the backup file
            # bpy.ops.wm.open_mainfile(filepath=str(backup_file))

            # Select the new object since we reloaded
            selected_object = bpy.data.objects.get(selected_mesh_name)

            # Assign the texture_final to the object
            self.assign_texture_to_object(
                selected_object, str(output_path / "final_texture.png")
            )

        return {"FINISHED"}

    def save_texture(self, texture, filepath):
        """
        Save a numpy array as an image texture in Blender.

        :param texture: numpy array representing the texture (shape: [height, width, channels]).
                        The array should be in the range [0, 255] for integer values.
        :param path: The path where the texture will be saved.
        """

        (height, width) = texture.shape[:2]

        # Ensure the numpy array is in float format and normalize if necessary
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255.0

        # Handle grayscale textures (add alpha channel if needed)
        if texture.shape[2] == 1:  # Grayscale
            texture = np.repeat(texture, 4, axis=2)
            texture[:, :, 3] = 1.0  # Set alpha to 1 for grayscale

        elif texture.shape[2] == 3:  # RGB
            alpha_channel = np.ones((height, width, 1), dtype=np.float32)
            texture = np.concatenate(
                (texture, alpha_channel), axis=2
            )  # Add alpha channel

        # Flatten the numpy array to a list
        flattened_texture = texture.flatten()

        # Create a new image in Blender with the provided dimensions
        image = bpy.data.images.new(
            name="SavedTexture", width=width, height=height, alpha=True
        )

        # Update the image's pixel data with the flattened texture
        image.pixels = flattened_texture

        # Save the image to the specified path
        image.filepath_raw = filepath
        image.file_format = "PNG"  # Set to PNG or any other format you prefer
        image.save()

    def load_texture(self, filepath):
        """Load a texture from a file and return it as a numpy array."""
        if os.path.exists(filepath):
            # Load the texture using Blender's image system
            image = bpy.data.images.load(filepath)

            # Get image dimensions
            width, height = image.size

            # Extract the pixel data (Blender stores it in RGBA format, float [0, 1])
            pixels = np.array(image.pixels[:], dtype=np.float32)

            # Reshape the flattened pixel data into (height, width, 4) array
            pixels = pixels.reshape((height, width, 4))

            # Convert the pixel values from float [0, 1] to [0, 255] uint8
            pixels = (pixels * 255).astype(np.uint8)

            return pixels
        else:
            raise FileNotFoundError(f"Texture file {filepath} not found.")

    def no_material_on(self, mesh):
        if 1 > len(mesh.materials):
            return True

        for i in range(len(mesh.materials)):
            if mesh.materials[i] is not None:
                return False
        return True

    def assign_texture_to_object(self, obj, texture_filepath):
        """
        Assign the final texture to the object's material using the texture filepath.

        :param obj: The object to assign the texture to.
        :param texture_filepath: Path to the texture file (PNG or other format).
        """
        # Check if the object has an existing material
        if obj.data.materials:
            material = obj.data.materials[0]
        else:
            # Create a new material if none exists
            material = bpy.data.materials.new(name="GeneratedMaterial")
            obj.data.materials.append(material)

        # Enable 'Use Nodes' for the material
        material.use_nodes = True
        nodes = material.node_tree.nodes

        # Find the Principled BSDF node or add one if it doesn't exist
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            nodes["Material Output"].location = (400, 0)

        # Create a new image texture node
        texture_node = nodes.new(type="ShaderNodeTexImage")

        # Load the texture file as an image
        try:
            texture_image = bpy.data.images.load(texture_filepath)
            texture_node.image = texture_image
        except RuntimeError as e:
            self.report({"ERROR"}, f"Could not load texture: {e}")
            return

        # Link the texture node to the Base Color input of the Principled BSDF node
        material.node_tree.links.new(
            bsdf.inputs["Base Color"], texture_node.outputs["Color"]
        )

        # Set the location of nodes for better layout
        texture_node.location = (-300, 0)
        bsdf.location = (0, 0)


class OBJECT_OT_SelectPipette(bpy.types.Operator):
    bl_idname = "object.select_pipette"
    bl_label = "Select Object with Pipette"

    def execute(self, context):
        context.scene.my_mesh_object = context.object.name
        return {"FINISHED"}
