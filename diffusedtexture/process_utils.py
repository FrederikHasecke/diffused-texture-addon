import math
import bpy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def blendercs_to_ccs(points_bcs, camera, rotation_only=False):
    """
    Converts a point cloud from the Blender coordinate system to the camera coordinate system.
    """
    # Extract camera rotation in world space
    camera_rotation = np.array(camera.matrix_world.to_quaternion().to_matrix()).T

    # Apply the rotation to align normals with the camera’s view
    if rotation_only:
        point_3d_cam = np.dot(camera_rotation, points_bcs.T).T
    else:
        # Translate points to the camera's coordinate system
        camera_position = np.array(camera.matrix_world.to_translation()).reshape((3,))
        points_bcs = points_bcs - camera_position
        point_3d_cam = np.dot(camera_rotation, points_bcs.T).T

    # Convert to camera coordinate system by inverting the Z-axis
    R_blender_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    point_3d_cam = np.dot(R_blender_to_cv, point_3d_cam.T).T

    return point_3d_cam


# TODO: Make this work like "create_texture_with_view_weighting.py"
def process_uv_texture(
    scene, uv_images, output_quad, target_resolution=512, render_resolution=2048
):
    """
    Processes the UV texture from multiple images and applies the inpainting on missing pixels.

    :param scene: Blender scene object, used to get the output path.
    :param uv_images: List of UV images corresponding to different camera views.
    :param output_quad: Output quad image containing the combined render.
    :param target_resolution: Resolution for each UV quadrant.
    :param render_resolution: Resolution of each render quadrant in the output_quad.
    :return: Inpainted UV texture.
    """

    num_cameras = len(uv_images)

    # Convert the output quad to a NumPy array and save it for debugging
    output_quad = np.array(output_quad)

    # Resize output_quad to render resolution
    output_quad = cv2.resize(
        output_quad,
        (
            int(output_quad.shape[0] * render_resolution / 512),
            int(output_quad.shape[0] * render_resolution / 512),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    resized_tiles = []
    for cam_index in range(num_cameras):
        # Calculate the position in the grid
        row = int((cam_index // int(math.sqrt(num_cameras))) * render_resolution)
        col = int((cam_index % int(math.sqrt(num_cameras))) * render_resolution)

        output_chunk = output_quad[
            row : row + render_resolution, col : col + render_resolution
        ]

        resized_tiles.append(output_chunk)

    # TODO: CONTINUE HERE !!!!! (2024-10-26)

    # Initialize the UV texture with NaNs to allow for averaging later
    uv_texture_first_pass = np.full(
        (num_cameras, target_resolution, target_resolution, 3), np.nan, dtype=np.float32
    )

    # Process each camera's UV image and populate the UV texture
    for cam_index in range(num_cameras):
        row = int((cam_index // int(math.sqrt(num_cameras))) * render_resolution)
        col = int((cam_index % int(math.sqrt(num_cameras))) * render_resolution)

        # Extract and mask the UV image content
        uv_image = uv_images[cam_index][..., :2]  # Keep only u and v channels
        uv_content = np.any(uv_image > 0, axis=-1)
        uv_image[~uv_content] = np.nan  # Mask invalid UV regions with NaN

        # Scale UV coordinates to target resolution
        uv_coordinates = (
            (uv_image * (target_resolution - 1)).astype(np.uint16).reshape(-1, 2)
        )
        uv_coordinates[:, 1] = (
            target_resolution - 1 - uv_coordinates[:, 1]
        )  # Flip the y-axis

        # Apply modulo operation to handle UV wrapping
        uv_coordinates = uv_coordinates % target_resolution

        # Populate the UV texture for each camera view
        uv_texture_first_pass[
            cam_index, uv_coordinates[:, 1], uv_coordinates[:, 0], :
        ] = output_quad[
            row : row + render_resolution, col : col + render_resolution
        ].reshape(
            -1, 3
        )

    # Compute the averaged UV texture while ignoring NaN values
    nan_mask = np.isnan(uv_texture_first_pass).all(axis=-1)
    unpainted_mask = np.all(nan_mask, axis=0).astype(np.uint8) * 255
    uv_texture_first_pass = np.nanmean(uv_texture_first_pass, axis=0).astype(np.uint8)

    # Inpaint missing pixels based on the unpainted mask
    filled_uv_texture_first_pass = cv2.inpaint(
        uv_texture_first_pass,
        unpainted_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )

    return filled_uv_texture_first_pass
