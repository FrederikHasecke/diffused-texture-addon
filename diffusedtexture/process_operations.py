import os
import math

import bpy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

from condition_setup import (
    bpy_img_to_numpy,
    create_depth_condition,
    create_normal_condition,
    create_similar_angle_image,
)

from render_setup import (
    setup_render_settings,
    create_cameras_on_sphere,
    create_cameras_on_one_ring,
)


def process_uv_texture(
    uv_images, facing_images, output_grid, target_resolution=512, render_resolution=2048
):
    """
    Processes the UV texture from multiple images and applies the inpainting on missing pixels.

    :param scene: Blender scene object, used to get the output path.
    :param uv_images: List of UV images corresponding to different camera views.
    :param output_grid: Output grid image containing the combined render.
    :param target_resolution: Resolution for each UV grid image.
    :param render_resolution: Resolution of each render in the output_grid.
    :return: Inpainted UV texture.
    """

    num_cameras = len(uv_images)

    # Convert the output grid to a NumPy array and save it for debugging
    output_grid = np.array(output_grid)

    # Resize output_grid to render resolution
    output_grid = cv2.resize(
        output_grid,
        (
            int(output_grid.shape[0] * render_resolution / 512),
            int(output_grid.shape[0] * render_resolution / 512),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    resized_tiles = []
    for cam_index in range(num_cameras):
        # Calculate the position in the grid
        row = int((cam_index // int(math.sqrt(num_cameras))) * render_resolution)
        col = int((cam_index % int(math.sqrt(num_cameras))) * render_resolution)

        output_chunk = output_grid[
            row : row + render_resolution, col : col + render_resolution
        ]

        resized_tiles.append(output_chunk)

    # create a 16x512x512x3 uv map (one for each grid img)
    uv_texture_first_pass = np.zeros(
        (num_cameras, target_resolution, target_resolution, 3), dtype=np.float32
    )

    # create a 16x512x512x3 uv map (one for each grid img)
    uv_texture_first_pass_weight = np.zeros(
        (num_cameras, target_resolution, target_resolution), dtype=np.float32
    )

    for cam_index in range(num_cameras):
        # Calculate the position in the grid
        row = int((cam_index // int(math.sqrt(num_cameras))) * render_resolution)
        col = int((cam_index % int(math.sqrt(num_cameras))) * render_resolution)

        # load the uv image
        uv_image = uv_images[cam_index]
        uv_image = uv_image[..., :2]  # Keep only u and v

        content_mask = np.zeros((render_resolution, render_resolution))
        content_mask[np.sum(uv_image, axis=-1) > 0] = 255
        content_mask = content_mask.astype(np.uint8)

        uv_image[content_mask == 0] = 0

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
        ] = resized_tiles[cam_index].reshape(-1, 3)

        uv_texture_first_pass_weight[
            cam_index, uv_coordinates[:, 1], uv_coordinates[:, 0], ...
        ] = facing_images[cam_index].reshape(
            -1,
        )

    # multiply the texture channels by the point at factor
    weighted_tex = (
        np.stack(
            (
                uv_texture_first_pass_weight,
                uv_texture_first_pass_weight,
                uv_texture_first_pass_weight,
            ),
            axis=-1,
        )
        * uv_texture_first_pass
    )

    weighted_tex = np.sum(weighted_tex, axis=0) / np.stack(
        (
            np.sum(uv_texture_first_pass_weight, axis=0),
            np.sum(uv_texture_first_pass_weight, axis=0),
            np.sum(uv_texture_first_pass_weight, axis=0),
        ),
        axis=-1,
    )

    # make the unpainted mask 0-255 uint8
    unpainted_mask = np.zeros((target_resolution, target_resolution))
    unpainted_mask[np.sum(uv_texture_first_pass_weight, axis=0) <= 0] = 255

    # inpaint all the missing pixels
    filled_uv_texture = cv2.inpaint(
        weighted_tex.astype(np.uint8),
        unpainted_mask.astype(np.uint8),
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )

    return filled_uv_texture


def load_image(base_path, index, frame_number, prefix=""):
    """Helper function to load an image file as a numpy array."""
    file_path = Path(base_path) / f"{prefix}camera_{index:02d}_{frame_number:04d}.exr"
    return bpy_img_to_numpy(str(file_path))[..., :3]


def load_depth_image(base_path, index, frame_number):
    """Helper function to load and process a depth image."""
    file_path = Path(base_path) / f"depth_camera_{index:02d}_{frame_number:04d}.exr"
    return create_depth_condition(str(file_path))[..., :3]


def load_normal_image(base_path, index, frame_number, camera):
    """Helper function to load and process a normal image."""
    file_path = Path(base_path) / f"normal_camera_{index:02d}_{frame_number:04d}.exr"
    return create_normal_condition(str(file_path), camera)[..., :3]


def generate_multiple_views(scene, max_size):
    """
    Generates texture maps for a 3D model using multiple views and saves outputs for depth, normal, UV, position, and image maps.

    :param scene: The Blender scene object.
    :param max_size: Maximum bounding box size of the model.
    :param create_cameras_on_one_ring: Function to create cameras on a single ring around the object.
    :param create_cameras_on_sphere: Function to create cameras in a spherical arrangement.
    :param setup_render_settings: Function to set up render settings and node outputs.
    :return: A dictionary of rendered images with keys: 'depth', 'normal', 'uv', 'position', 'image', 'facing'.
    """

    # Set parameters
    num_cameras = int(scene.num_cameras)
    render_resolution = 2048

    # Create cameras based on the number specified in the scene
    if num_cameras == 4:
        cameras = create_cameras_on_one_ring(
            num_cameras=num_cameras, max_size=max_size, name_prefix="Camera_first_pass"
        )
    elif num_cameras in [9, 16]:
        cameras = create_cameras_on_sphere(
            num_cameras=num_cameras, max_size=max_size, name_prefix="Camera_first_pass"
        )
    else:
        raise ValueError("Only 4, 9, or 16 cameras are supported.")

    # Set up render nodes and paths
    output_path = Path(scene.output_path)
    output_nodes = setup_render_settings(
        scene, resolution=(render_resolution, render_resolution)
    )
    output_dirs = ["depth", "normal", "uv", "position", "img"]
    for output_type in output_dirs:
        output_nodes[output_type].base_path = str(
            output_path / f"first_pass_{output_type}"
        )
        os.makedirs(output_nodes[output_type].base_path, exist_ok=True)

    # Update to make new cameras available
    bpy.context.view_layer.update()

    # Initialize lists for loaded images
    (
        depth_images,
        normal_images,
        uv_images,
        position_images,
        img_images,
        facing_images,
    ) = ([] for _ in range(6))
    frame_number = scene.frame_current

    # Render and load images for each camera
    for i, camera in enumerate(cameras):
        scene.camera = camera  # Set active camera

        # Set file paths for each render pass
        for pass_type in output_dirs:
            output_nodes[pass_type].file_slots[0].path = f"{pass_type}_camera_{i:02d}_"

        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)  # Render and save images

        # Load UV image
        uv_images.append(
            load_image(output_nodes["uv"].base_path, i, frame_number, "uv_")
        )

        # Load depth image with processing
        depth_images.append(
            load_depth_image(output_nodes["depth"].base_path, i, frame_number)
        )

        # Load position image
        position_images.append(
            load_image(output_nodes["position"].base_path, i, frame_number, "position_")
        )

        # Load normal image with processing
        normal_images.append(
            load_normal_image(output_nodes["normal"].base_path, i, frame_number, camera)
        )

        # Create a facing ratio image to show alignment between normals and camera direction
        facing_img = create_similar_angle_image(
            load_image(output_nodes["normal"].base_path, i, frame_number, "normal_")[
                ..., :3
            ],
            position_images[-1][..., :3],
            camera,
        )
        facing_images.append(facing_img)

    return {
        "depth": depth_images,
        "normal": normal_images,
        "uv": uv_images,
        "position": position_images,
        "image": img_images,
        "facing": facing_images,
    }


def assemble_multiview_grid(
    multiview_images, render_resolution=2048, sd_resolution=512
):
    """
    Assembles images from multiple views into a structured grid, applies a mask, and resizes the outputs.

    :param multiview_images: Dictionary containing lists of images for 'depth', 'normal', 'facing', and 'uv'.
    :param target_resolution: Target resolution for the output image.
    :param render_resolution: Resolution for rendering each camera view before scaling.
    :return: A dictionary containing assembled grids for 'depth', 'normal', 'uv', 'facing', and 'content mask'.
    """

    num_cameras = len(multiview_images["depth"])
    grid_size = int(math.sqrt(num_cameras))  # Assuming a square grid

    # Initialize empty arrays for each type of grid
    grids = initialize_grids(grid_size, render_resolution)

    # Populate the grids with multiview images
    for i, (depth_img, normal_img, facing_img, uv_img) in enumerate(
        zip(
            multiview_images["depth"],
            multiview_images["normal"],
            multiview_images["facing"],
            multiview_images["uv"],
        )
    ):
        row, col = compute_grid_position(i, grid_size, render_resolution)
        populate_grids(
            grids,
            depth_img,
            normal_img,
            facing_img,
            uv_img,
            row,
            col,
            render_resolution,
        )

    # Generate content mask and input image
    grids["content_mask"] = create_content_mask(grids["uv_grid"])

    # Resize grids to target resolution for SD model input
    resized_grids = resize_grids(grids, render_resolution, sd_resolution)

    # Create the canny for the resized grids
    resized_grids["canny_grid"] = np.stack(
        (
            cv2.Canny(resized_grids["normal_grid"].astype(np.uint8), 100, 200),
            cv2.Canny(resized_grids["normal_grid"].astype(np.uint8), 100, 200),
            cv2.Canny(resized_grids["normal_grid"].astype(np.uint8), 100, 200),
        ),
        axis=-1,
    )

    return grids, resized_grids


def initialize_grids(grid_size, render_resolution):
    """Initialize blank grids for each map type."""
    grid_shape = (grid_size * render_resolution, grid_size * render_resolution)
    return {
        "depth_grid": np.zeros((*grid_shape, 3), dtype=np.uint8),
        "normal_grid": np.zeros((*grid_shape, 3), dtype=np.uint8),
        "facing_grid": np.zeros(grid_shape, dtype=np.uint8),
        "uv_grid": np.zeros((*grid_shape, 3), dtype=np.float32),
    }


def compute_grid_position(index, grid_size, render_resolution):
    """Compute row and column position in the grid based on index."""
    row = (index // grid_size) * render_resolution
    col = (index % grid_size) * render_resolution
    return row, col


def populate_grids(
    grids, depth_img, normal_img, facing_img, uv_img, row, col, render_resolution
):
    """Populate each grid with the corresponding multiview images."""
    grids["depth_grid"][
        row : row + render_resolution, col : col + render_resolution
    ] = depth_img
    grids["normal_grid"][
        row : row + render_resolution, col : col + render_resolution
    ] = normal_img
    grids["facing_grid"][
        row : row + render_resolution, col : col + render_resolution
    ] = (255 * facing_img).astype(np.uint8)
    grids["uv_grid"][
        row : row + render_resolution, col : col + render_resolution
    ] = uv_img


def create_content_mask(uv_img):
    """Generate a content mask from the UV image."""
    content_mask = np.zeros(uv_img.shape[:2], dtype=np.uint8)
    content_mask[np.sum(uv_img, axis=-1) > 0] = 255
    return cv2.dilate(content_mask, np.ones((5, 5), np.uint8), iterations=3)


def resize_grids(grids, render_resolution, sd_resolution):
    """Resize grids to target resolution for Stable Diffusion model."""
    scale_factor = render_resolution / sd_resolution
    resized_grids = {}
    for key, grid in grids.items():
        interpolation = cv2.INTER_NEAREST if "mask" in key else cv2.INTER_LINEAR
        resized_grids[key] = cv2.resize(
            grid, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation
        )
    return resized_grids


def save_multiview_grids(multiview_grids, output_folder, file_format="png"):
    """
    Save each grid in multiview_grids to the specified output folder.

    :param multiview_grids: Dictionary where keys are grid names and values are images (numpy arrays).
    :param output_folder: Path to the output folder where images will be saved.
    :param file_format: File format for the saved images (default is 'png').
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    for key, image in multiview_grids.items():
        file_path = os.path.join(output_folder, f"{key}.{file_format}")

        # Convert to BGR format if needed for OpenCV (assumes input is RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write the image to disk
        cv2.imwrite(file_path, image)

        print(f"Saved {key} grid to {file_path}")
