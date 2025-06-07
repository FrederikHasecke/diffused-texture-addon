import cv2
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


def save_debug_images(
    scene,
    i,
    input_image_sd,
    content_mask_render_sd,
    content_mask_texture,
    canny_img,
    normal_img,
    depth_img,
    output,
):
    """Save intermediate debugging images."""
    output_path = scene.output_path
    cv2.imwrite(
        f"{output_path}/input_image_sd_{i}.png",
        cv2.cvtColor(input_image_sd, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(f"{output_path}/content_mask_render_sd_{i}.png", content_mask_render_sd)
    cv2.imwrite(f"{output_path}/content_mask_texture_{i}.png", content_mask_texture)
    cv2.imwrite(
        f"{output_path}/canny_{i}.png", cv2.cvtColor(canny_img, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        f"{output_path}/normal_{i}.png", cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        f"{output_path}/depth_{i}.png", cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        f"{output_path}/output_{i}.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    )


def isolate_object(obj):
    """Temporarily hide all other objects and move this one to origin."""
    hidden_objects = []

    for other in bpy.data.objects:
        if other != obj:
            if not other.hide_get():
                other.hide_set(True)
                hidden_objects.append(other)

            if not other.hide_render:
                other.hide_render = True

    # Save original transform
    original_location = obj.location.copy()

    # Move to origin
    obj.location = (0.0, 0.0, 0.0)

    return {
        "hidden_objects": hidden_objects,
        "target_object": obj,
        "original_location": original_location,
    }


def restore_scene(backup_data):
    """Restore object position and re-show hidden objects."""

    obj = backup_data["target_object"]
    obj.location = backup_data["original_location"]

    for o in backup_data["hidden_objects"]:
        o.hide_set(False)
        o.hide_render = False
