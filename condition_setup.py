import os
import bpy
import numpy as np
from diffusedtexture.process_utils import blendercs_to_ccs


def bpy_img_to_numpy(img_path):
    """
    Load an image as a Blender image and converts it to a NumPy array.

    :param img_path: The path to the image.
    :return: A NumPy array representation of the image.
    """

    # Load image using Blender's bpy
    bpy.data.images.load(img_path)

    # Get the file name from the path
    img_file_name = os.path.basename(img_path)

    # Access the image by name after loading
    img_bpy = bpy.data.images.get(img_file_name)

    # Get image dimensions
    width, height = img_bpy.size

    # Determine the number of channels
    num_channels = len(img_bpy.pixels) // (width * height)

    # Convert the flat pixel array to a NumPy array
    pixels = np.array(img_bpy.pixels[:], dtype=np.float32)

    # Reshape the array to match the image's dimensions and channels
    image_array = pixels.reshape((height, width, num_channels))

    # FLIP vertically
    image_array = np.flipud(image_array)

    return image_array


def create_depth_condition(depth_image_path):

    depth_array = bpy_img_to_numpy(depth_image_path)[..., 0]

    # Replace large values with NaN (assuming 1e10 represents invalid depth)
    depth_array[depth_array >= 1e10] = np.nan

    # Invert the depth values so that closer objects have higher values
    depth_array = np.nanmax(depth_array) - depth_array

    # Normalize the depth array to range [0, 1]
    depth_array -= np.nanmin(depth_array)
    depth_array /= np.nanmax(depth_array)

    # Add a small margin to the background
    depth_array += 10 / 255.0  # Approximately 0.039

    # Replace NaN values with 0 (background)
    depth_array[np.isnan(depth_array)] = 0

    # Normalize again to ensure values are within [0, 1]
    depth_array /= np.nanmax(depth_array)

    # Scale to [0, 255] and convert to uint8
    depth_array = (depth_array * 255).astype(np.uint8)

    # stack on third axis to get a rgb png
    depth_array = np.stack((depth_array, depth_array, depth_array), axis=-1)

    return depth_array


def create_normal_condition(normal_img_path, camera_obj):
    normal_array = bpy_img_to_numpy(normal_img_path)

    normal_array = normal_array[..., :3]

    # Get image dimensions
    image_size = normal_array.shape[:2]

    # Flatten the normal array for transformation
    normal_pc = normal_array.reshape((-1, 3))

    # Rotate the normal vectors to the camera space without translating
    normal_pc = blendercs_to_ccs(
        points_bcs=normal_pc, camera=camera_obj, rotation_only=True
    )

    # Map normalized values to the [0, 1] range for RGB display
    red_channel = ((normal_pc[:, 0] + 1) / 2).reshape(image_size)  # Normal X
    green_channel = ((normal_pc[:, 1] + 1) / 2).reshape(image_size)  # Normal Y
    blue_channel = ((normal_pc[:, 2] + 1) / 2).reshape(image_size)  # Normal Z

    # Adjust to shapenet colors
    blue_channel = 1 - blue_channel
    green_channel = 1 - green_channel

    # Stack channels into a single image
    normal_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    normal_image = np.clip(normal_image, 0, 1)

    # Convert to uint8 for display
    normal_image *= 255.0
    normal_image = normal_image.astype(np.uint8)

    return normal_image
