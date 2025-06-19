import math
import os
import shutil
from pathlib import Path
from typing import Any

import bpy
import cv2
import numpy as np
from numpy.typing import NDArray

from ..blender_operations import load_img_to_numpy
from ..condition_setup import (
    create_depth_condition,
    create_normal_condition,
)


def delete_render_folders(render_img_folders: list) -> None:
    for render_folder in render_img_folders:
        # Check if the folder exists
        if Path(render_folder).exists() and Path(render_folder).is_dir():
            # Delete the folder and all its contents
            shutil.rmtree(render_folder)


def process_uv_texture(  # noqa: PLR0913
    context: bpy.types.Context,
    uv_images: list[NDArray],
    facing_images: list[NDArray],
    output_grid: NDArray,
    target_resolution: int = 512,
    render_resolution: int = 2048,
    facing_percentile: float = 1.0,
) -> NDArray:
    num_cameras = len(uv_images)

    if context.scene.custom_sd_resolution:
        sd_resolution = int(
            int(context.scene.custom_sd_resolution)
            // np.sqrt(int(context.scene.num_cameras)),
        )
    else:
        sd_resolution = 512 if context.scene.sd_version == "sd15" else 1024

    # Resize output_grid to render resolution
    output_grid = cv2.resize(
        output_grid,
        (
            int(
                (output_grid.shape[0] * render_resolution / sd_resolution),
            ),
            int(
                (output_grid.shape[0] * render_resolution / sd_resolution),
            ),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    resized_tiles = []
    for cam_index in range(num_cameras):
        # Calculate the position in the grid
        row = int((cam_index // int(math.sqrt(num_cameras))) * render_resolution)
        col = int((cam_index % int(math.sqrt(num_cameras))) * render_resolution)

        output_chunk = output_grid[
            row : row + render_resolution,
            col : col + render_resolution,
        ]

        resized_tiles.append(output_chunk)

    # create a 16x512x512x3 uv map (one for each grid img)
    uv_texture_first_pass = np.zeros(
        (num_cameras, target_resolution, target_resolution, 3),
        dtype=np.float32,
    )

    # create a 16x512x512x3 uv map (one for each grid img)
    uv_texture_first_pass_weight = np.zeros(
        (num_cameras, target_resolution, target_resolution),
        dtype=np.float32,
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

        # the uvs are meant to start from the bottom left corner
        # so we flip the y axis (v axis)
        uv_coordinates[:, 1] = int(target_resolution - 1) - uv_coordinates[:, 1]

        # in case we have uv coordinates beyond the texture
        uv_coordinates = uv_coordinates % int(target_resolution)

        uv_texture_first_pass[
            cam_index,
            uv_coordinates[:, 1],
            uv_coordinates[:, 0],
            ...,
        ] = resized_tiles[cam_index].reshape(-1, 3)

        # adjust the facing weight to the chosen percentile
        cur_facing_image = facing_images[cam_index]

        # goes from 0 to 1
        # we cut of the bottom 1.0-facing_percentile and stretch the rest 0 to 1
        cur_facing_image = cur_facing_image * (
            1.0 + facing_percentile
        )  # 0..1 now 0..1.2
        cur_facing_image = cur_facing_image - facing_percentile  # 0..1.2 now -0.2..1.0
        cur_facing_image[cur_facing_image < 0] = 0

        uv_texture_first_pass_weight[
            cam_index,
            uv_coordinates[:, 1],
            uv_coordinates[:, 0],
            ...,
        ] = cur_facing_image.reshape(
            -1,
        )

    return inpaint_missing(
        target_resolution,
        uv_texture_first_pass,
        uv_texture_first_pass_weight,
    )


def inpaint_missing(
    target_resolution: int,
    uv_texture_first_pass: NDArray,
    uv_texture_first_pass_weight: NDArray,
    lower_bound: float = 0.1,
) -> NDArray:
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
    unpainted_mask[np.sum(uv_texture_first_pass_weight, axis=0) <= lower_bound] = 255

    # inpaint all the missing pixels
    return cv2.inpaint(
        weighted_tex.astype(np.uint8),
        unpainted_mask.astype(np.uint8),
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )


def load_image(
    base_path: str,
    index: int,
    frame_number: int,
    prefix: str = "",
) -> NDArray[Any]:
    """Helper function to load an image file as a numpy array."""
    file_path = Path(base_path) / f"{prefix}camera_{index:02d}_{frame_number:04d}.exr"
    return load_img_to_numpy(str(file_path))[..., :3]


def load_depth_image(
    base_path: str,
    index: int,
    frame_number: int,
) -> NDArray[Any]:
    """Helper function to load and process a depth image."""
    file_path = Path(base_path) / f"depth_camera_{index:02d}_{frame_number:04d}.exr"
    return create_depth_condition(str(file_path))[..., :3]


def load_normal_image(
    base_path: str,
    index: int,
    frame_number: int,
    camera: bpy.types.Camera,
) -> NDArray[Any]:
    """Helper function to load and process a normal image."""
    file_path = Path(base_path) / f"normal_camera_{index:02d}_{frame_number:04d}.exr"
    return create_normal_condition(str(file_path), camera)[..., :3]


def assemble_multiview_grid(
    texture: NDArray[np.uint8] | None,
    multiview_images: dict[str, str],
    render_resolution: int = 2048,
    sd_resolution: int = 512,
) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Assemble images from multiple views into a structured grid.

    Args:
        multiview_images (dict[str, list]): Dictionary containing lists of images for
                                            'depth', 'normal', 'facing', and 'uv'.
        render_resolution (int, optional): Resolution for rendering each camera view
                                            before scaling. Defaults to 2048.
        sd_resolution (int, optional): Target resolution for the SD images.
                                        Defaults to 512.

    Returns:
        tuple[dict[str, Any], dict]: Contains assembled grids for 'depth', 'normal',
                                    'uv', 'facing', and 'content mask'.

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
            strict=False,
        ),
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

    # TODO(Frederik): Use facing img as img input

    # Generate content mask and input image
    grids["content_grid"] = create_content_mask(grids["uv_grid"])

    # Create the input image grid from the texture (if provided)
    grids["input_grid"] = create_input_image_grid(
        texture, grids["uv_grid"], grids["content_grid"]
    )

    # Resize grids to target resolution for SD model input
    resized_grids = resize_grids(grids, render_resolution, sd_resolution)

    # Overwrite the content and canny grid with newly created resized versions
    resized_grids["content_grid"] = create_content_mask(resized_grids["uv_grid"])
    resized_grids["canny_grid"] = np.stack(
        (
            cv2.Canny(resized_grids["normal_grid"].astype(np.uint8), 100, 200),
            cv2.Canny(resized_grids["normal_grid"].astype(np.uint8), 100, 200),
            cv2.Canny(resized_grids["normal_grid"].astype(np.uint8), 100, 200),
        ),
        axis=-1,
    )
    resized_grids["input_grid"] = create_input_image_grid(
        texture, resized_grids["uv_grid"], resized_grids["content_grid"]
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


def compute_grid_position(
    index: int,
    grid_size: int,
    render_resolution: int,
) -> tuple[int, int]:
    """Compute row and column position in the grid based on index."""
    row = (index // grid_size) * render_resolution
    col = (index % grid_size) * render_resolution
    return row, col


def populate_grids(
    grids: dict[str, NDArray],
    depth_img: NDArray,
    normal_img: NDArray,
    facing_img: NDArray,
    uv_img: NDArray,
    row: int,
    col: int,
    render_resolution: int,
) -> None:
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
    grids["uv_grid"][row : row + render_resolution, col : col + render_resolution] = (
        uv_img
    )


def create_content_mask(uv_img):
    """Generate a content mask from the UV image."""
    content_mask = np.zeros(uv_img.shape[:2], dtype=np.uint8)
    content_mask[np.sum(uv_img, axis=-1) > 0] = 255
    return cv2.dilate(content_mask, np.ones((10, 10), np.uint8), iterations=3)


def resize_grids(
    grids, render_resolution, sd_resolution, interpolation=cv2.INTER_NEAREST
):
    """Resize grids to target resolution for Stable Diffusion model."""

    scale_factor = sd_resolution / render_resolution
    resized_grids = {}
    for key, grid in grids.items():
        height, width = grid.shape[:2]

        interpolation = interpolation if "mask" in key else cv2.INTER_LINEAR
        resized_grids[key] = cv2.resize(
            grid,
            (int(scale_factor * height), int(scale_factor * width)),
            interpolation=interpolation,
        )
    return resized_grids


def save_multiview_grids(multiview_grids, output_folder, file_format="png", prefix=""):
    """
    Save each grid in multiview_grids to the specified output folder.

    :param multiview_grids: Dictionary where keys are grid names and values are images (numpy arrays).
    :param output_folder: Path to the output folder where images will be saved.
    :param file_format: File format for the saved images (default is 'png').
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    for key, image in multiview_grids.items():
        file_path = os.path.join(output_folder, f"{prefix}{key}.{file_format}")

        if image.dtype == np.float32:
            image = 255 * image
            image = image.astype(np.uint8)

        # Convert to BGR format if needed for OpenCV (assumes input is RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write the image to disk
        cv2.imwrite(file_path, image)

        print(f"Saved {key} grid to {file_path}")


def create_input_image(texture, uv, render_resolution, texture_resolution):
    """
    Project a texture onto renderings using UV coordinates, then resize the projected texture
    images to match the resolution of the resized UV rendering images.

    :param texture: Input texture image (H, W, C) with texture resolution as dimensions.
    :param uv: Original UV coordinates (H, W, 2) corresponding to the renderings, values in range [0, 1].
    :param input_texture_res: Target Resolution for the input
    :param render_resolution: render_resolution
    :param texture_resolution: texture_resolution
    :return: The projected and resized texture as 3 image arrays.
    """

    input_texture_res = texture.shape[0]

    # Scale UV coordinates to texture coordinates
    uv_scaled = uv[..., :2]
    uv_scaled = (uv_scaled * (input_texture_res - 1)).astype(np.int32)
    uv_scaled[..., 1] = int(input_texture_res - 1) - uv_scaled[..., 1]

    # Reshape UV coordinates for indexing
    uv_coordinates = uv_scaled[..., :2].reshape(-1, 2)
    uv_coordinates = uv_coordinates % input_texture_res  # Handle wrap-around

    # Ensure texture has three channels
    texture = texture[..., :3]

    # Project texture using the UV coordinates
    projected_texture = texture[uv_coordinates[:, 1], uv_coordinates[:, 0], :3].reshape(
        (uv.shape[0], uv.shape[1], 3)
    )

    # Resize to match the resized UV rendering dimensions
    input_texture_resolution = cv2.resize(
        projected_texture,
        (input_texture_res, input_texture_res),
        interpolation=cv2.INTER_LINEAR,
    )

    input_render_resolution = cv2.resize(
        projected_texture,
        (render_resolution, render_resolution),
        interpolation=cv2.INTER_LINEAR,
    )

    target_texture_resolution = cv2.resize(
        projected_texture,
        (texture_resolution, texture_resolution),
        interpolation=cv2.INTER_LINEAR,
    )

    return (
        input_texture_resolution,
        input_render_resolution,
        target_texture_resolution,
    )


def create_input_image_grid(
    texture: NDArray[np.uint8] | None,
    uv_grid: NDArray[np.float32],
    content_mask: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Project a texture onto renderings using UV coordinates."""
    if texture is None:
        return np.ones_like(uv_grid) * 255

    input_texture_res = texture.shape[0]

    # Scale UV coordinates to texture coordinates
    uv_scaled: NDArray[np.float32] = uv_grid[..., :2]

    # 0-1 range to 0-input_texture_res range
    uv_scaled = (uv_scaled * (input_texture_res - 1)).astype(np.int32)

    # Flip the y-axis (v-axis) to match texture coordinates
    uv_scaled[..., 1] = int(input_texture_res - 1) - uv_scaled[..., 1]

    # Reshape UV coordinates for indexing
    uv_coordinates = uv_scaled[..., :2].reshape(-1, 2)
    uv_coordinates = (
        uv_coordinates % input_texture_res
    )  # Handle wrap-around UV coordinates

    # Ensure texture has three channels (remove alpha if present)
    texture = texture[..., :3]

    projected_area = np.ones((uv_grid.shape[0], uv_grid.shape[1]), dtype=np.uint8)
    projected_area[uv_coordinates[:, 1], uv_coordinates[:, 0]] = 0

    projected_texture_grid = np.zeros(
        (uv_grid.shape[0], uv_grid.shape[1], 3),
        dtype=np.uint8,
    )

    # Project texture using the UV coordinates
    projected_texture_grid[uv_coordinates[:, 1], uv_coordinates[:, 0], :3] = texture[
        uv_coordinates[:, 1], uv_coordinates[:, 0], :3
    ]

    # interpolate missing pixels (projected_area == 1)
    inpainted_texture = cv2.inpaint(
        projected_texture_grid,
        projected_area,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )

    if content_mask is not None:
        # Apply content mask to the inpainted texture
        inpainted_texture[content_mask == 0] = 255

    return inpainted_texture.astype(np.uint8)
