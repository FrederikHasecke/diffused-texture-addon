import bpy
import numpy as np


def update_uv_maps(self, context):
    """
    Update the list of available UV maps for the selected mesh object.
    This function is used in the panel to populate the UV map dropdown.
    """
    obj = bpy.data.objects.get(self.my_mesh_object)
    if obj and obj.type == "MESH":
        uv_layers = obj.data.uv_layers.keys()
        return [(uv, uv, "") for uv in uv_layers]
    else:
        return [("None", "None", "")]


def get_mesh_objects(self, context):
    """
    Get all mesh objects in the current Blender scene.
    This function is used to populate the mesh object dropdown in the panel.
    """
    return [(obj.name, obj.name, "") for obj in bpy.data.objects if obj.type == "MESH"]


def update_image_list(self, context):
    """
    Update the list of available images in the Blender file.
    This function is used to populate the image selection dropdown in the panel.
    """
    images = [(img.name, img.name, "") for img in bpy.data.images]
    if not images:
        images.append(("None", "None", "No images available"))
    return images


def apply_texture_to_uv_map(obj, uv_map_name, image_name):
    """
    Apply the selected image to the specified UV map of the selected object.
    This is used when the user selects an image in the panel and it needs to be
    applied to the object's UV map.
    """
    image = bpy.data.images.get(image_name)
    if image:
        for uv_layer in obj.data.uv_layers:
            if uv_layer.name == uv_map_name:
                # Assuming the object has a material and texture slots
                material = obj.material_slots[0].material
                texture_slot = material.texture_paint_slots[0]
                texture_slot.texture.image = image
                break


def image_to_numpy(image):
    """
    Convert a Blender image object to a NumPy array.

    :param image: Blender image object
    :return: NumPy array representing the image (height, width, channels)
    """
    # Get image dimensions
    width, height = image.size

    # Get pixel data (Blender stores pixels in a flat array as RGBA values in [0, 1])
    pixels = np.array(image.pixels[:], dtype=np.float32)

    # Reshape to (height, width, 4) since Blender stores data in RGBA format
    pixels = pixels.reshape((height, width, 4))

    # Discard the alpha channel
    pixels = pixels[:, :, :3]  # Keep only RGB channels

    # Convert the pixel values from [0, 1] to [0, 255] for typical image use
    pixels = (pixels * 255).astype(np.uint8)

    return pixels
