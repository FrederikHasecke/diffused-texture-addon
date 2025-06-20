import bpy
import numpy as np
from numpy.typing import NDArray

from ..blender_operations import load_img_to_numpy
from .process_utils import blendercs_to_ccs


def create_depth_condition(depth_image_path: str, invalid_depth: int = 1e10) -> NDArray:
    depth_array = load_img_to_numpy(depth_image_path)[..., 0]

    # Replace large invalid values with NaN
    depth_array[depth_array >= invalid_depth] = np.nan

    # Invert the depth values so that closer objects have higher values
    depth_array = np.nanmax(depth_array) - depth_array

    # Normalize the depth array to range [0, 1]
    depth_array -= np.nanmin(depth_array)
    depth_array /= np.nanmax(depth_array)

    # Add a small margin to the background
    depth_array += 10 / 255.0  # Approximately 0.039

    # normalize
    depth_array[np.isnan(depth_array)] = 0
    depth_array /= np.nanmax(depth_array)
    depth_array = (depth_array * 255).astype(np.uint8)

    return np.stack((depth_array, depth_array, depth_array), axis=-1)


def create_normal_condition(
    normal_img_path: str,
    camera_obj: bpy.types.Object,
) -> NDArray:
    normal_array = load_img_to_numpy(normal_img_path)

    normal_array = normal_array[..., :3]

    # Get image dimensions
    image_size = normal_array.shape[:2]

    # Flatten the normal array for transformation
    normal_pc = normal_array.reshape((-1, 3))

    # Rotate the normal vectors to the camera space without translating
    normal_pc = blendercs_to_ccs(
        points_bcs=normal_pc,
        camera=camera_obj,
        rotation_only=True,
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

    return (normal_image * 255.0).astype(np.uint8)
