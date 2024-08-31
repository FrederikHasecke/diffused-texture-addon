import os
import bpy
import cv2
import numpy as np

from render_setup import setup_render_settings, create_cameras_on_sphere
from condition_setup import (
    bpy_img_to_numpy,
    create_depth_condition,
    create_normal_condition,
)


def first_pass(scene, max_size):
    """Run the first pass for texture generation."""

    # create 16 equidistant cameras
    cameras = create_cameras_on_sphere(
        num_cameras=16,
        max_size=max_size,  # we dont want to intersect with the object
        name_prefix="Camera_first_pass",
        offset=False,
    )

    # TODO: Do we want to enable more than 512x512?
    output_nodes = setup_render_settings(scene, resolution=(512, 512))

    # Set base paths for outputs
    output_nodes["depth"].base_path = scene.output_path + "first_pass_depth/"
    output_nodes["normal"].base_path = scene.output_path + "first_pass_normal/"
    output_nodes["uv"].base_path = scene.output_path + "first_pass_uv/"
    output_nodes["position"].base_path = scene.output_path + "first_pass_position/"

    # Create directories if they don't exist
    for key in output_nodes:
        os.makedirs(output_nodes[key].base_path, exist_ok=True)

    # Render with each of the cameras in the list
    for i, camera in enumerate(cameras):
        # Set the active camera to the current camera
        scene.camera = camera

        # Set the output file path for each pass with the corresponding camera index
        output_nodes["depth"].file_slots[0].path = f"depth_camera_{i:02d}_"
        output_nodes["normal"].file_slots[0].path = f"normal_camera_{i:02d}_"
        output_nodes["uv"].file_slots[0].path = f"uv_camera_{i:02d}_"
        output_nodes["position"].file_slots[0].path = f"position_camera_{i:02d}_"

        # Render the scene
        bpy.ops.render.render(write_still=True)

    # Load the rendered images
    depth_images = []
    normal_images = []
    uv_images = []
    position_images = []

    for i in range(16):

        # Load UV images
        uv_image_path = f"{output_nodes['uv'].base_path}uv_camera_{i:02d}_0000.exr"
        uv_image_bpy = bpy.data.images.load(uv_image_path)
        uv_image = bpy_img_to_numpy(uv_image_bpy)

        uv_images.append(uv_image)

        # Load position images
        position_image_path = (
            f"{output_nodes['position'].base_path}position_camera_{i:02d}_0000.exr"
        )
        position_image_bpy = bpy.data.images.load(position_image_path)

        position_image = bpy_img_to_numpy(position_image_bpy)
        position_images.append(position_image)

        # Load depth images
        depth_image_path = (
            f"{output_nodes['depth'].base_path}depth_camera_{i:02d}_0000.exr"
        )

        depth_image_bpy = bpy.data.images.load(depth_image_path)
        depth_image = create_depth_condition(depth_image_bpy)

        depth_images.append(depth_image)

        # Load normal images
        normal_image_path = (
            f"{output_nodes['normal'].base_path}normal_camera_{i:02d}_0000.exr"
        )

        normal_image_bpy = bpy.data.images.load(normal_image_path)
        normal_image = create_normal_condition(normal_image_bpy)
        normal_images.append(normal_image)

    # create the empty grid images
    depth_quad = np.zeros((2048, 2048, 3), dtype=np.uint8)
    normal_quad = np.zeros((2048, 2048, 3), dtype=np.uint8)

    # Combine the 16 images into a 4x4 grid structure
    for depth_img, normal_img in zip(depth_images, normal_images):
        # Calculate the position in the grid
        row = (i // 4) * 512
        col = (i % 4) * 512

        # Add the images to the grid
        depth_quad[row : row + 512, col : col + 512] = depth_img
        normal_quad[row : row + 512, col : col + 512] = normal_img

    canny_quad = cv2.Canny(normal_quad, 100, 200)
    canny_quad = np.stack((canny_quad, canny_quad, canny_quad), axis=-1)

    # TODO: Create the Canny image from the normal

    # TODO: Setup the controlnets according to the mesh complexity

    # TODO: Create the pipe with the optional addition of a new checkpoint and additional loras

    # TODO: Create the corresponding list of images for the controlnets

    # TODO: Execute pipe

    # TODO: Reproject the 4x4 img output grid to the UV positions iteratively each on a channel of a 16 channel image

    # TODO: nanmean the colors
