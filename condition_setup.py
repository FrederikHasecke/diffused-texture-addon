import os
import bpy
import numpy as np
from texturegen.process_utils import blendercs_to_ccs


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


def create_normal_condition(normal_img_path, position_img_path, camera_obj):
    normal_array = bpy_img_to_numpy(normal_img_path)
    position_array = bpy_img_to_numpy(position_img_path)

    # Get image dimensions
    image_size = normal_array.shape[:2]

    # transform the position array into a pointcloud
    normal_pc = normal_array[..., :3].reshape((-1, 3))
    position_pc = position_array[..., :3].reshape((-1, 3))

    # rotate (and translate) the normal and position to camera space
    normal_pc = blendercs_to_ccs(
        points_bcs=normal_pc, camera=camera_obj, rotation_only=True
    )
    position_pc = blendercs_to_ccs(points_bcs=position_pc, camera=camera_obj)

    # normalize the norm and pos pc
    # Ignore div by zero and div by nan warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        normal_pc_norm = normal_pc / np.linalg.norm(normal_pc, axis=1, keepdims=True)
        position_pc_norm = position_pc / np.linalg.norm(
            position_pc, axis=1, keepdims=True
        )

    red_channel = (normal_pc_norm[..., 0] + position_pc_norm[..., 0]).reshape(
        image_size
    )  # actually minus pos, since reverse vector

    # invert and normalize
    red_channel -= -2  # 0 to max
    red_channel /= 4  # 0 to 1
    red_channel = 1 - red_channel  # invert

    green_channel = (normal_pc_norm[..., 1] + position_pc_norm[..., 1]).reshape(
        image_size
    )  # actually minus pos, since reverse vector

    green_channel -= -2  # 0 to max
    green_channel /= 4  # 0 to 1
    green_channel = 1 - green_channel  # invert

    # blue channel: get the z pointing direction
    blue_channel = np.linalg.norm(
        np.cross(normal_pc_norm, position_pc_norm, axis=-1), axis=-1
    ).reshape(image_size)
    blue_channel = 1 - blue_channel  # invert (0 to 1)
    blue_channel = 0.5 + 0.5 * blue_channel  # shapenet normal z is 128-255

    normal_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

    normal_image *= 255.0
    normal_image = normal_image.astype(np.uint8)

    return normal_image
