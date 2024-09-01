import os
import bpy
import cv2
import numpy as np
from pathlib import Path


from render_setup import setup_render_settings, create_cameras_on_sphere
from condition_setup import (
    bpy_img_to_numpy,
    create_depth_condition,
    create_normal_condition,
)
from texturegen.diffusers_utils import (
    create_first_pass_pipeline,
    infer_first_pass_pipeline,
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

    output_path = Path(scene.output_path)  # Use Path for OS-independent path handling

    # TODO: Do we want to enable more than 512x512?
    output_nodes = setup_render_settings(scene, resolution=(512, 512))

    # Set base paths for outputs
    output_nodes["depth"].base_path = str(output_path / "first_pass_depth")
    output_nodes["normal"].base_path = str(output_path / "first_pass_normal")
    output_nodes["uv"].base_path = str(output_path / "first_pass_uv")
    output_nodes["position"].base_path = str(output_path / "first_pass_position")

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

    # get the current blender frame
    frame_number = bpy.context.scene.frame_current

    for i in range(16):

        # Load UV images
        uv_image_path = (
            Path(output_nodes["uv"].base_path)
            / f"uv_camera_{i:02d}_{frame_number:04d}.exr"
        )

        uv_image = bpy_img_to_numpy(str(uv_image_path))

        uv_images.append(uv_image)

        # Load depth images
        depth_image_path = (
            Path(output_nodes["depth"].base_path)
            / f"depth_camera_{i:02d}_{frame_number:04d}.exr"
        )

        depth_image = create_depth_condition(str(depth_image_path))

        depth_images.append(depth_image)

        # Load position images
        position_image_path = (
            Path(output_nodes["position"].base_path)
            / f"position_camera_{i:02d}_{frame_number:04d}.exr"
        )
        # Load normal images
        normal_image_path = (
            Path(output_nodes["normal"].base_path)
            / f"normal_camera_{i:02d}_{frame_number:04d}.exr"
        )
        normal_image = create_normal_condition(
            str(normal_image_path), str(position_image_path), cameras[i]
        )
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

    # Create the Canny image from the normal
    canny_quad = cv2.Canny(normal_quad, 100, 200)
    canny_quad = np.stack((canny_quad, canny_quad, canny_quad), axis=-1)

    # TODO: Create the pipe with the optional addition of a new checkpoint and additional loras
    pipe = create_first_pass_pipeline(scene)
    output_quad = infer_first_pass_pipeline(
        pipe, scene, canny_quad, normal_quad, depth_quad
    )

    output_quad = np.array(output_quad)

    # create a 16x512x512x3 uv map (one for each grid img)
    uv_texture_first_pass = np.nan * np.ones(
        (16, 512, 512, 3), dtype=np.float32
    )  # neccessary for nanmean

    for cam_index in enumerate(cameras):
        # Calculate the position in the grid
        row = (i // 4) * 512
        col = (i % 4) * 512

        # load the uv image
        uv_image = uv_images[cam_index]

        # resize the uv values to 0-511
        uv_coordinates = (uv_image * 511).astype(np.uint16).reshape(-1, 2)

        # the uvs are meant to start from the bottom left corner, so we flip the y axis (v axis)
        uv_coordinates[:, 1] = 511 - uv_coordinates[:, 1]

        uv_texture_first_pass[i, uv_coordinates[:, 1], uv_coordinates[:, 0], ...] = (
            output_quad[row : row + 512, col : col + 512].reshape(-1, 3)
        )

    # average the UV texture across the 16 quadrants, but only where the UV map is defined
    nan_per_channel = np.any(
        np.isnan(uv_texture_first_pass), axis=-1
    )  # check if any nan values are present in the last axis (channels)
    nan_per_image = np.all(
        nan_per_channel, axis=0
    )  # check if any nan values are present in the last axis (quadrants)

    # now every pixel in nan_per_image is True if the pixel is nan in all of the 4 quadrants
    # make the unpainted mask 0-255 uint8
    unpainted_mask = nan_per_image.astype(np.uint8) * 255

    uv_texture_first_pass = np.nanmean(uv_texture_first_pass, axis=0).astype(np.uint8)

    # inpaint all the missing pixels
    filled_uv_texture_first_pass = cv2.inpaint(
        uv_texture_first_pass,
        unpainted_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )

    return filled_uv_texture_first_pass
