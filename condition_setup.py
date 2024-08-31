import numpy as np
from texturegen.process_utils import blendercs_to_ccs


def bpy_img_to_numpy(img_bpy):
    """
    Converts a Blender image to a NumPy array.

    :param image: The Blender image object (bpy.types.Image).
    :return: A NumPy array representation of the image.
    """

    # Ensure the image has been loaded and has pixel data
    if not img_bpy.has_data:
        raise ValueError("The image has no data loaded.")

    # Get image dimensions
    width, height = img_bpy.size

    # Determine the number of channels
    num_channels = len(img_bpy.pixels) // (width * height)

    # Convert the flat pixel array to a NumPy array
    pixels = np.array(img_bpy.pixels[:], dtype=np.float32)

    # Reshape the array to match the image's dimensions and channels
    image_array = pixels.reshape(height, width, num_channels)

    return image_array


def create_depth_condition(depth_image_bpy):

    depth_array = bpy_img_to_numpy(depth_image_bpy)[..., 0]

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

    return depth_array



def create_normal_condition(normal_img_bpy, position_img_bpy, camera_obj):
    normal_array = bpy_img_to_numpy(normal_img_bpy)
    position_array = bpy_img_to_numpy(position_img_bpy)

    # transform the position array into a pointcloud
    position_pc = 




def create_normal_image_conditioning(output_dir, conditionings_dir, camera_info, frame):
    """
    This function creates the normal image conditioning.

    Args:
        output_dir (str): The directory to save the normal image conditioning to.
        camera_info (dict): The camera information.
        frame (int): The frame to create the normal image conditioning for.

    Returns:
        None

    Raises:
        FileNotFoundError: If the normal image does not exist.
    """

    # Search pattern for the normal exr image
    normal_image_search_pattern = os.path.join(
        output_dir, "normal_{:04d}.exr".format(frame)
    )
    position_image_search_pattern = os.path.join(
        output_dir, "position_{:04d}.exr".format(frame)
    )

    # Check if the images exist
    if not os.path.exists(normal_image_search_pattern):
        raise FileNotFoundError(
            "The normal image {} does not exist.".format(normal_image_search_pattern)
        )
    if not os.path.exists(position_image_search_pattern):
        raise FileNotFoundError(
            "The normal image {} does not exist.".format(position_image_search_pattern)
        )

    # Load the images
    normal_image = openexr_numpy.imread(normal_image_search_pattern)
    position_image = openexr_numpy.imread(position_image_search_pattern)

    # get the original image size
    image_size = normal_image.shape[:2]

    # transform the position and normal images to nx3 arrays
    normal_image = normal_image.reshape((-1, 3))
    position_image = position_image.reshape((-1, 3))

    # rotate the normal image to camera space
    normal_pc = simis.utility.GeneralUtility.blendercs_2_ccs_rot_only(
        normal_image, camera_info
    )
    position_pc = simis.utility.GeneralUtility.blendercs_2_ccs(
        position_image, camera_info
    )

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
    blue_channel = 0.5 + 0.5 * blue_channel

    normal_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

    normal_image *= 255.0
    normal_image = normal_image.astype(np.uint8)

    # Save the angle conditioning image as a png to the conditionings directory
    angle_conditioning_path = os.path.join(
        conditionings_dir, "normal_{:04d}.png".format(frame)
    )
    imageio.imwrite(angle_conditioning_path, normal_image)
