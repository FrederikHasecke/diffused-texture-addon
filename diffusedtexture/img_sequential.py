try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from collections.abc import Callable
from typing import cast

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from numpy.typing import NDArray

from ..blender_operations import ProcessParameter
from ..model_support import get_default_sd_resolution
from .pipeline.pipeline_builder import create_diffusion_pipeline

try:
    from .pipeline.pipeline_runner import TextureGenerationCancelledError, run_pipeline
except ImportError:
    from .pipeline.pipeline_runner import run_pipeline

    class TextureGenerationCancelledError(RuntimeError):
        """Fallback cancellation error for tests with a stubbed runner."""


from .process_operations import (
    _require_cv2,
    assemble_multiview_list,
    blend_texture_with_alpha,
    smooth_alpha_map,
)

if cv2 is not None:
    CV2_INTER_NEAREST = cv2.INTER_NEAREST
    CV2_INTER_LINEAR = cv2.INTER_LINEAR
    CV2_INTER_LANCZOS4 = cv2.INTER_LANCZOS4
else:
    # OpenCV enum integer values; used only until deps are installed
    CV2_INTER_NEAREST = 0
    CV2_INTER_LINEAR = 1
    CV2_INTER_LANCZOS4 = 4

ALPHA_PAINT_THRESHOLD = 1e-3


def img_sequential(  # noqa: PLR0913
    multiview_images: dict[str, list[NDArray]],
    process_parameter: ProcessParameter,
    progress_callback: Callable,
    should_cancel: Callable[[], bool] = lambda: False,
    texture: NDArray[np.float32] | None = None,
    facing_percentile: float = 0.5,
) -> NDArray[np.uint8]:
    """Process multiview images sequentially view-by-view with diffusion pipeline.

    Args:
        multiview_images: Dict of multiview input images per view.
        process_parameter: Processing parameters.
        progress_callback: Function to track progress.
        should_cancel: Returns True when generation should stop.
        texture: Optional texture to project.
        facing_percentile: Float value for facing percentile.

    Returns:
        Processed full-resolution texture.
    """
    _require_cv2()
    sd_resolution = get_default_sd_resolution(
        process_parameter.sd_version,
        process_parameter.custom_sd_resolution,
    )

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

    # TODO: Only inpaint for 2-3 pixels for each view, and paste onto texture with feathering  # noqa: E501
    # TODO: Check if para/weighted approach on all views improve the texture later on

    texres = int(process_parameter.texture_resolution)
    unpainted_mask = 255 * np.ones((texres, texres), dtype=np.uint8)
    # Initialize previous_texture before the loop
    previous_texture = (
        (255 * texture).astype(np.uint8)
        if texture is not None
        else 255 * np.ones_like(resized_list["input"][0], dtype=np.uint8)
    )
    for i in range(n_views):
        if should_cancel():
            msg = "Cancelled by user."
            raise TextureGenerationCancelledError(msg)

        arr_in = resized_list["input"][i]
        arr_content = resized_list["content"][i]
        arr_uv = resized_list["uv"][i]

        if i == 0:
            input_img = Image.fromarray(arr_in.astype(np.uint8))
            mask_img = Image.fromarray(arr_content)
        else:
            current_unpainted_mask = cast("NDArray[np.uint8]", unpainted_mask)
            input_img, _ = create_new_view_input(
                previous_texture,
                current_unpainted_mask,
                arr_in,
                arr_uv,
            )

            facing_uint = resized_list["facing"][i]

            arr_content[facing_uint < (facing_percentile * 255)] = 0

            mask_img = Image.fromarray(arr_content)

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
            should_cancel=should_cancel,
            strength=process_parameter.denoise_strength,
            guidance_scale=process_parameter.guidance_scale,
            num_inference_steps=process_parameter.num_inference_steps,
        )

        if should_cancel():
            msg = "Cancelled by user."
            raise TextureGenerationCancelledError(msg)

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

    if texture is None:
        # inpaint the texture for all areas that are not covered by the views
        # if no texture is provided, we assume the texture is white
        previous_texture = cv2.inpaint(
            previous_texture.astype(np.uint8),
            unpainted_mask.astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )

    return cast("NDArray[np.uint8]", previous_texture)


def project_view_to_texture(  # noqa: PLR0913
    sd_result: Image,  # type: ignore  # noqa: PGH003
    uv_view: NDArray[np.uint8],
    facing_view: NDArray[np.uint8],
    texture_resolution: int,
    texture: NDArray[np.uint8],
    unpainted_mask: NDArray[np.uint8] | None = None,
    facing_percentile: float = 0.5,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None]:
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
            interpolation=CV2_INTER_NEAREST,
        )
        facing_scaled = cv2.resize(
            facing_scaled,
            (texture_resolution, texture_resolution),
            interpolation=CV2_INTER_NEAREST,
        )

        new_texture[facing_scaled > 0] = texture_scaled[facing_scaled > 0]
        facing_texture[facing_scaled > 0] = facing_scaled[facing_scaled > 0]

    alpha = (facing_texture.astype(np.float32) / 255.0) * (1.0 + facing_percentile)
    alpha = np.clip(alpha - facing_percentile, 0.0, 1.0)
    alpha = smooth_alpha_map(alpha)
    alpha *= (facing_texture > 0).astype(np.float32)

    if unpainted_mask is not None:
        unpainted_mask[alpha > ALPHA_PAINT_THRESHOLD] = 0

    existing_texture = np.zeros_like(new_texture)
    if texture is not None:
        # Ensure texture is in the correct format, else resize it
        if texture.ndim == 3:  # noqa: PLR2004
            if texture.shape[2] == 4:  # noqa: PLR2004
                texture = texture[..., :3]
            texture = cv2.resize(texture, (texture_resolution, texture_resolution))
        existing_texture = texture[..., :3]

    # TODO(Frederik): Change the approach to stack all sequential textures
    # Inpaint missing areas on each and stack them together with the facing percentiles
    # as weighting.

    return blend_texture_with_alpha(
        existing_texture,
        new_texture,
        alpha,
    ), unpainted_mask


def create_new_view_input(
    output_texture: NDArray[np.uint8],
    unpainted_mask: NDArray[np.uint8],
    input_view: NDArray[np.uint8],
    uv_view: NDArray[np.uint8],
) -> tuple[Image, NDArray[np.uint8]]:  # type: ignore  # noqa: PGH003
    """Create new input image for the current view by projecting the output texture."""
    # get the uv coordinates from the uv_view
    uv_view = uv_view[..., :2]  # Ensure UVs are 2D
    h, w = input_view.shape[0], input_view.shape[1]
    tex_h, tex_w = output_texture.shape[0], output_texture.shape[1]

    uv_scaled = np.empty_like(uv_view)
    uv_scaled[..., 0] = uv_view[..., 0] * (tex_w - 1)
    uv_scaled[..., 1] = uv_view[..., 1] * (tex_h - 1)

    uv_y = ((tex_h - 1) - uv_scaled[..., 1].flatten().astype(int)) % tex_h
    uv_x = uv_scaled[..., 0].flatten().astype(int) % tex_w

    # Project the output texture to the input view using UV coordinates
    view_from_texture = output_texture[uv_y, uv_x]
    view_from_texture = view_from_texture.reshape(h, w, 3)
    mask_from_texture = unpainted_mask[uv_y, uv_x]
    mask_from_texture = mask_from_texture.reshape(h, w)
    mask_from_texture = np.stack([mask_from_texture] * 3, axis=-1)  # Make it 3-channel

    # mask_from_texture is either 0 or 255, where 255 == unpainted
    mask_from_texture[mask_from_texture == 255] = 1  # noqa: PLR2004

    return Image.fromarray(view_from_texture), mask_from_texture[..., 0]
