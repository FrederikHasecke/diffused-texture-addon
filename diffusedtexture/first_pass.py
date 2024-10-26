import os
import math
import bpy
import cv2
import numpy as np
from pathlib import Path


from render_setup import (
    setup_render_settings,
    create_cameras_on_sphere,
    create_cameras_on_two_rings,
    create_cameras_on_one_ring,
)
from condition_setup import (
    bpy_img_to_numpy,
    create_depth_condition,
    create_normal_condition,
    create_similar_angle_image,
)
from diffusedtexture.diffusers_utils import (
    create_first_pass_pipeline,
    infer_first_pass_pipeline,
)
from diffusedtexture.process_utils import process_uv_texture


def first_pass(scene, max_size):
    """Run the first pass for texture generation."""

    num_cameras = int(scene.num_cameras)

    # target_resolution = int(2 * end_resolution)
    target_resolution = int(scene.texture_resolution)

    render_resolution = 2048  # TODO: Test if this prevents the dittering effect

    if num_cameras == 4:
        cameras = create_cameras_on_one_ring(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="Camera_first_pass",
        )
    elif num_cameras == 9:
        cameras = create_cameras_on_sphere(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="Camera_first_pass",
            offset=False,
        )
    elif num_cameras == 16:
        cameras = create_cameras_on_two_rings(
            num_cameras=num_cameras, max_size=max_size, name_prefix="Camera_first_pass"
        )

    else:
        raise ValueError("Only 4, 9 or 16 cameras are supported for first pass.")

    output_path = Path(scene.output_path)  # Use Path for OS-independent path handling

    output_nodes = setup_render_settings(
        scene, resolution=(render_resolution, render_resolution)
    )

    # Set base paths for outputs
    output_nodes["depth"].base_path = str(output_path / "first_pass_depth")
    output_nodes["normal"].base_path = str(output_path / "first_pass_normal")
    output_nodes["uv"].base_path = str(output_path / "first_pass_uv")
    output_nodes["position"].base_path = str(output_path / "first_pass_position")
    output_nodes["img"].base_path = str(output_path / "first_pass_img")

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
        output_nodes["img"].file_slots[0].path = f"img_camera_{i:02d}_"

        # Render the scene
        bpy.ops.render.render(write_still=True)

    # Load the rendered images
    depth_images = []
    normal_images = []
    position_images = []
    uv_images = []
    img_images = []
    facing_images = []

    # get the current blender frame
    frame_number = bpy.context.scene.frame_current

    for i, camera in enumerate(cameras):

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

        position_image = bpy_img_to_numpy(str(position_image_path))
        position_images.append(position_image)

        # Load normal images
        normal_image_path = (
            Path(output_nodes["normal"].base_path)
            / f"normal_camera_{i:02d}_{frame_number:04d}.exr"
        )
        normal_image = create_normal_condition(str(normal_image_path), camera)
        normal_images.append(normal_image)

        # Load render images
        img_image_path = (
            Path(output_nodes["img"].base_path)
            / f"img_camera_{i:02d}_{frame_number:04d}.exr"
        )

        img_image = bpy_img_to_numpy(str(img_image_path))

        img_images.append(img_image)

        # create the facing images
        facing_img = create_similar_angle_image(normal_image, position_image, camera)

        facing_images.append(facing_img)

    # create the empty grid images
    depth_quad = np.zeros(
        (
            int(math.sqrt(num_cameras) * render_resolution),
            int(math.sqrt(num_cameras) * render_resolution),
            3,
        ),
        dtype=np.uint8,
    )
    normal_quad = np.zeros(
        (
            int(math.sqrt(num_cameras) * render_resolution),
            int(math.sqrt(num_cameras) * render_resolution),
            3,
        ),
        dtype=np.uint8,
    )
    img_quad = np.zeros(
        (
            int(math.sqrt(num_cameras) * render_resolution),
            int(math.sqrt(num_cameras) * render_resolution),
            3,
        ),
        dtype=np.uint8,
    )

    # Combine the 16 images into a 4x4 grid structure
    # TODO: CHECK IF THIS FIXES THE ASSIGNMENT!
    for i, (depth_img, normal_img, img_img) in enumerate(
        zip(depth_images, normal_images, img_images)
    ):
        # Calculate the position in the grid
        row = int((i // math.sqrt(num_cameras)) * render_resolution)
        col = int((i % math.sqrt(num_cameras)) * render_resolution)

        # Add the images to the grid
        depth_quad[row : row + render_resolution, col : col + render_resolution] = (
            depth_img
        )
        normal_quad[row : row + render_resolution, col : col + render_resolution] = (
            normal_img
        )

        img_quad[row : row + render_resolution, col : col + render_resolution] = (
            img_img[..., :3] / np.max(img_img[..., :3]) * 255
        ).astype(np.uint8)

    # Create the Canny image from the normal
    canny_quad = cv2.Canny(img_quad, 100, 200)
    canny_quad = np.stack((canny_quad, canny_quad, canny_quad), axis=-1)

    # Input image full white image
    content_mask = np.zeros(
        (
            int(math.sqrt(num_cameras) * render_resolution),
            int(math.sqrt(num_cameras) * render_resolution),
        )
    )
    content_mask[np.sum(normal_quad, axis=-1) > 0] = 255
    content_mask = content_mask.astype(np.uint8)

    bigger_content_mask = cv2.dilate(
        content_mask, np.ones((5, 5), np.uint8), iterations=2
    )

    smaller_content_mask = cv2.erode(
        content_mask, np.ones((5, 5), np.uint8), iterations=1
    )

    input_image = 255 * np.ones(
        (
            int(math.sqrt(num_cameras) * render_resolution),
            int(math.sqrt(num_cameras) * render_resolution),
            3,
        )
    )
    # input_image[content_mask == 0] = 255
    # input_image = np.clip(input_image, 0, 255)
    input_image = input_image.astype(np.uint8)

    """
        upscale_texture = cv2.resize(
        texture,
        (target_resolution, target_resolution),
        interpolation=cv2.INTER_LANCZOS4,
    )
    """

    # Resize the images to a 512 square content size
    input_image_sd = cv2.resize(
        input_image,
        (
            int(input_image.shape[0] / render_resolution * 512),
            int(input_image.shape[0] / render_resolution * 512),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    bigger_content_mask_sd = cv2.resize(
        bigger_content_mask,
        (
            int(bigger_content_mask.shape[0] / render_resolution * 512),
            int(bigger_content_mask.shape[0] / render_resolution * 512),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    canny_quad_sd = cv2.resize(
        canny_quad,
        (
            int(canny_quad.shape[0] / render_resolution * 512),
            int(canny_quad.shape[0] / render_resolution * 512),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    normal_quad_sd = cv2.resize(
        normal_quad,
        (
            int(normal_quad.shape[0] / render_resolution * 512),
            int(normal_quad.shape[0] / render_resolution * 512),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    depth_quad_sd = cv2.resize(
        depth_quad,
        (
            int(depth_quad.shape[0] / render_resolution * 512),
            int(depth_quad.shape[0] / render_resolution * 512),
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    # TODO: Create the pipe with the optional addition of a new checkpoint and additional loras
    pipe = create_first_pass_pipeline(scene)
    output_quad = infer_first_pass_pipeline(
        pipe,
        scene,
        input_image_sd,
        bigger_content_mask_sd,
        canny_quad_sd,
        normal_quad_sd,
        depth_quad_sd,
        strength=scene.denoise_strength,
    )

    cv2.imwrite(
        os.path.join(scene.output_path, "input_image.png"),
        cv2.cvtColor(input_image_sd, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, "content_mask.png"),
        cv2.cvtColor(bigger_content_mask_sd, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, "canny_quad.png"),
        cv2.cvtColor(canny_quad_sd, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, "normal_quad.png"),
        cv2.cvtColor(normal_quad_sd, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, "depth_quad.png"),
        cv2.cvtColor(depth_quad_sd, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, "img_quad.png"),
        cv2.cvtColor(img_quad, cv2.COLOR_RGB2BGR),
    )

    output_quad = np.array(output_quad)

    # TODO: REMOVE ME, just for debugging
    # output the image as png
    cv2.imwrite(
        os.path.join(scene.output_path, "output_quad.png"),
        cv2.cvtColor(output_quad, cv2.COLOR_RGB2BGR),
    )

    # resize output to render resolution
    output_quad = cv2.resize(
        output_quad,
        (
            int(output_quad.shape[0] * render_resolution / 512),
            int(output_quad.shape[0] * render_resolution / 512),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # create a 16x512x512x3 uv map (one for each grid img)
    uv_texture_first_pass = np.nan * np.ones(
        (num_cameras, target_resolution, target_resolution, 3), dtype=np.float32
    )  # neccessary for nanmean

    for cam_index in range(len(cameras)):
        # Calculate the position in the grid
        row = int((cam_index // int(math.sqrt(num_cameras))) * render_resolution)
        col = int((cam_index % int(math.sqrt(num_cameras))) * render_resolution)

        # load the uv image
        uv_image = uv_images[cam_index]
        uv_image = uv_image[..., :2]  # Keep only u and v

        # cut the part out of the smaller_content_mask
        uv_content = smaller_content_mask[
            row : row + render_resolution, col : col + render_resolution
        ]
        uv_image[uv_content == 0] = np.nan

        # resize the uv values to 0-511
        uv_coordinates = (
            (uv_image * int(target_resolution - 1)).astype(np.uint16).reshape(-1, 2)
        )

        # the uvs are meant to start from the bottom left corner, so we flip the y axis (v axis)
        uv_coordinates[:, 1] = int(target_resolution - 1) - uv_coordinates[:, 1]

        # in case we have uv coordinates beyond the texture
        uv_coordinates = uv_coordinates % int(target_resolution)

        uv_texture_first_pass[
            cam_index, uv_coordinates[:, 1], uv_coordinates[:, 0], ...
        ] = output_quad[
            row : row + render_resolution, col : col + render_resolution
        ].reshape(
            -1, 3
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
