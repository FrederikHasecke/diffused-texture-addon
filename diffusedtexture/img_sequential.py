import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..blender_operations import ProcessParameter
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import assemble_multiview_list


def img_sequential(
    multiview_images: dict[str, list[NDArray]],
    process_parameter: ProcessParameter,
    progress_callback: callable,
    texture: NDArray[np.float32] | None = None,
    facing_percentile: float = 0.5,
) -> NDArray[np.float32]:
    """Process multiview images sequentially view-by-view with diffusion pipeline.

    Args:
        multiview_images: Dict of multiview input images per view.
        process_parameter: Processing parameters.
        progress_callback: Function to track progress.
        texture: Optional texture to project.
        facing_percentile: Float value for facing percentile.

    Returns:
        Processed full-resolution texture.
    """
    if process_parameter.custom_sd_resolution:
        sd_resolution = int(process_parameter.custom_sd_resolution)
    else:
        sd_resolution = 512 if process_parameter.sd_version == "sd15" else 1024

    # Assemble per-view processed/resized maps
    _, resized_list = assemble_multiview_list(
        texture=texture,
        multiview_images=multiview_images,
        sd_resolution=sd_resolution,
    )

    n_views = int(process_parameter.num_cameras)

    # Create the diffusion pipeline once
    pipeline = create_diffusion_pipeline(process_parameter)

    # create a sub callback to report progress
    def sub_progress_callback(sub_percent: int) -> None:
        percent = int((sub_percent + i * 100) / n_views)
        progress_callback(percent)

    keep_mask = None
    texres = int(process_parameter.texture_resolution)
    unpainted_mask = 255 * np.ones((texres, texres), dtype=np.uint8)
    for i in range(n_views):
        arr_in = resized_list["input"][i]
        arr_content = resized_list["content"][i]
        arr_uv = resized_list["uv"][i]

        if i == 0:
            input_img = Image.fromarray(arr_in.astype(np.uint8))
            mask_img = Image.fromarray(arr_content)
            previous_texture = (
                (255 * texture).astype(np.uint8)
                if texture is not None
                else 255 * np.ones_like(arr_in, dtype=np.uint8)
            )
        else:
            input_img, keep_mask = create_new_view_input(
                previous_texture,
                unpainted_mask,
                arr_in,
                arr_uv,
            )
            mask_img = Image.fromarray(arr_content)
            mask_img[resized_list["facing"][i] < facing_percentile * 255] = 0

        # Run the pipeline for the current view
        result = run_pipeline(
            pipe=pipeline,
            process_parameter=process_parameter,
            input_img=input_img,
            uv_mask=mask_img,
            canny_img=Image.fromarray(resized_list["canny"][i]),
            normal_img=Image.fromarray(resized_list["normal"][i]),
            depth_img=Image.fromarray(resized_list["depth"][i]),
            progress_callback=sub_progress_callback,
            strength=process_parameter.denoise_strength,
            guidance_scale=process_parameter.guidance_scale,
            num_inference_steps=process_parameter.num_inference_steps,
        )

        # Project the current result back to the texture
        previous_texture, unpainted_mask = project_view_to_texture(
            sd_result=result,
            uv_view=resized_list["uv"][i],
            facing_view=resized_list["facing"][i],
            texture_resolution=texres,
            texture=previous_texture,
            unpainted_mask=unpainted_mask,
            facing_percentile=facing_percentile,
        )

        # save all images with the index in the filename for debugging
        input_img.save(f"{process_parameter.output_path}/input_view_{i}.png")
        mask_img.save(f"{process_parameter.output_path}/mask_view_{i}.png")
        result.save(f"{process_parameter.output_path}/result_view_{i}.png")
        if hasattr(previous_texture, "save"):
            previous_texture.save(
                f"{process_parameter.output_path}/previous_texture_{i}.png",
            )
        else:
            # If previous_texture is ndarray, convert to Image
            Image.fromarray(previous_texture.astype(np.uint8)).save(
                f"{process_parameter.output_path}/previous_texture_{i}.png",
            )
        cv2.imwrite(
            f"{process_parameter.output_path}/unpainted_mask_{i}.png",
            unpainted_mask,
        )
        if isinstance(mask_img, Image.Image):
            cv2.imwrite(
                f"{process_parameter.output_path}/mask_img_{i}.png",
                np.array(mask_img),
            )
        else:
            cv2.imwrite(f"{process_parameter.output_path}/mask_img_{i}.png", mask_img)

    if texture is None:
        # inpaint the texture for all areas that are not covered by the views
        # if no texture is provided, we assume the texture is white
        previous_texture = cv2.inpaint(
            previous_texture.astype(np.uint8),
            unpainted_mask.astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )

    return previous_texture


def project_view_to_texture(  # noqa: PLR0913
    sd_result: Image.Image,
    uv_view: NDArray[np.uint8],
    facing_view: NDArray[np.uint8],
    texture_resolution: int,
    texture: NDArray[np.uint8],
    unpainted_mask: NDArray[np.uint8] = None,
    facing_percentile: float = 0.5,
) -> NDArray[np.uint8]:
    """Project the output of the diffusion model back onto the texture."""
    sd_array = np.array(sd_result)
    sd_array = sd_array[..., :3]  # Ensure we only take RGB channels

    uv_view = (
        uv_view[..., :2] * texture_resolution
    ) % texture_resolution  # handle UV wrapping

    # Clamp UVs to valid range
    uv_y = uv_view[..., 1].astype(int)
    uv_y = texture_resolution - 1 - uv_y  # Flip Y axis for correct orientation
    uv_x = uv_view[..., 0].astype(int)

    # Create a new texture to hold the projected result
    new_texture = np.zeros((texture_resolution, texture_resolution, 3), dtype=np.uint8)
    facing_texture = np.zeros((texture_resolution, texture_resolution), dtype=np.uint8)

    for scale in [0.25, 0.5, 1.0]:
        texture_scaled = np.zeros(
            (int(scale * texture_resolution), int(scale * texture_resolution), 3),
            dtype=np.uint8,
        )
        facing_scaled = np.zeros(
            (int(scale * texture_resolution), int(scale * texture_resolution)),
            dtype=np.uint8,
        )

        # Scale the UV coordinates
        scaled_uv_x = (uv_x * scale).astype(int)
        scaled_uv_y = (uv_y * scale).astype(int)

        # Project the output texture to the input view using UV coordinates
        texture_scaled[scaled_uv_y, scaled_uv_x] = sd_array
        facing_scaled[scaled_uv_y, scaled_uv_x] = facing_view

        # resize to target resolution
        texture_scaled = cv2.resize(
            texture_scaled,
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_NEAREST,
        )
        facing_scaled = cv2.resize(
            facing_scaled,
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_NEAREST,
        )

        new_texture[facing_scaled > 0] = texture_scaled[facing_scaled > 0]
        facing_texture[facing_scaled > 0] = facing_scaled[facing_scaled > 0]

    # keep areas which are facing the camera
    mask = facing_texture > (facing_percentile * 255)

    new_texture[~mask] = 0
    if unpainted_mask is not None:
        unpainted_mask[mask] = 0

    if texture is not None:
        # Ensure texture is in the correct format, else resize it
        if texture.ndim == 3:  # noqa: PLR2004
            if texture.shape[2] == 4:  # noqa: PLR2004
                texture = texture[..., :3]
            texture = cv2.resize(texture, (texture_resolution, texture_resolution))

        # Blend the new texture with the existing texture
        new_texture[~mask] = texture[..., :3][~mask]

    # TODO(Frederik): Change the approach to stack all sequential textures  # noqa: E501, FIX002
    # Inpaint missing areas on each and stack them together with the facing percentiles
    # as weighting.

    return new_texture, unpainted_mask


def create_new_view_input(
    output_texture: NDArray[np.float32],
    unpainted_mask: NDArray[np.uint8],
    input_view: NDArray[np.uint8],
    uv_view: NDArray[np.uint8],
) -> tuple[Image.Image, NDArray[np.uint8]]:
    """Create new input image for the current view by projecting the output texture."""
    # get the uv coordinates from the uv_view
    uv_view = uv_view[..., :2]  # Ensure UVs are 2D
    h, w = input_view.shape[0], input_view.shape[1]
    uv_view = uv_view * (w - 1) % w  # Scale UVs to input size

    # Clamp UVs to valid range
    uv_y = np.clip(uv_view[..., 1].flatten().astype(int), 0, h - 1)
    uv_x = np.clip(uv_view[..., 0].flatten().astype(int), 0, w - 1)

    # Project the output texture to the input view using UV coordinates
    view_from_texture = output_texture[uv_y, uv_x]
    view_from_texture = view_from_texture.reshape(h, w, 3)
    mask_from_texture = unpainted_mask[uv_y, uv_x]
    mask_from_texture = mask_from_texture.reshape(h, w)
    mask_from_texture = np.stack([mask_from_texture] * 3, axis=-1)  # Make it 3-channel

    # mask_from_texture is either 0 or 255, where 255 == unpainted
    mask_from_texture[mask_from_texture == 255] = 1  # noqa: PLR2004

    return Image.fromarray(view_from_texture), mask_from_texture[..., 0]
