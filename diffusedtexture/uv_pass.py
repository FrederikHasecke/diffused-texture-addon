from collections.abc import Callable
from pathlib import Path

import bpy

from .process_operations import _require_cv2

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import numpy as np
from numpy.typing import NDArray

from ..blender_operations import (
    ProcessParameter,
    load_img_to_numpy,
)
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline


def export_uv_layout(
    obj_name: str,
    export_path: str,
    uv_map_name: str | None = None,
    size: tuple[int, int] = (1024, 1024),
) -> None:
    """Export the UV layout of the mesh object and specified UV map to an image file.

    :param obj_name: Name of the object in Blender.
    :param export_path: File path where the UV layout should be saved.
    :param uv_map_name: Name of the UV map to use (default is the active UV map).
    :param size: Resolution of the UV layout image (default is 1024x1024).
    """
    # Get the object
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH":
        msg = f"Object '{obj_name}' not found or is not a mesh."
        raise ValueError(msg)

    # Set the object as active
    bpy.context.view_layer.objects.active = obj

    # Ensure it is in object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Get the UV map layers
    uv_layers = obj.data.uv_layers
    if not uv_layers:
        msg = f"No UV maps found for object {obj_name}."
        raise ValueError(msg)

    # Find or set the active UV map
    if uv_map_name:
        uv_layer = uv_layers.get(uv_map_name)
        if uv_layer is None:
            msg = f"UV map {uv_map_name} not found on object {obj_name}."
            raise ValueError(msg)
        uv_layers.active = uv_layer  # Set the selected UV map as active
    else:
        uv_layer = uv_layers.active  # Use active UV map if none is provided

    if uv_layer is None:
        msg = f"No active UV map found for object {obj_name}."
        raise ValueError(msg)

    # Set UV export settings
    bpy.ops.uv.export_layout(
        filepath=export_path,
        size=size,
        opacity=1.0,  # Adjust opacity of the lines (0.0 to 1.0)
        export_all=False,  # Export only active UV map
    )


def uv_pass(
    baked_texture_dict: dict[str, NDArray[np.float32]],
    process_parameter: ProcessParameter,
    progress_callback: Callable,
    texture_input: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Run the UV space refinement pass."""
    _require_cv2()

    obj_name = process_parameter.my_mesh_object
    uv_map_name = process_parameter.my_uv_map

    # uv output path
    uv_map_path = Path(
        process_parameter.output_path,
    )  # Use Path for OS-independent path handling
    uv_map_path = str(uv_map_path / "uv_map_layout.png")

    export_uv_layout(
        obj_name,
        uv_map_path,
        uv_map_name=uv_map_name,
        size=(
            int(process_parameter.texture_resolution),
            int(process_parameter.texture_resolution),
        ),
    )

    # load the uv layout
    uv_layout_image = load_img_to_numpy(str(uv_map_path))

    uv_layout_image = uv_layout_image / np.max(uv_layout_image)
    uv_layout_image = uv_layout_image * 255
    uv_layout_image = uv_layout_image.astype(np.uint8)

    # Texture mask based on the alpha channel (3rd channel in RGBA image)
    mask = uv_layout_image[..., 3]

    # Apply dilation to the mask
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    # Stack the mask into 3 channels (to match the RGB format)
    mask = np.stack((mask, mask, mask), axis=-1)
    mask = mask.astype(np.uint8)

    # Create the Canny edge detection mask
    canny = 255 * np.ones(
        (
            int(process_parameter.texture_resolution),
            int(process_parameter.texture_resolution),
        ),
    )

    # Only process areas where the alpha channel is 255
    canny[uv_layout_image[..., 3] >= 1] = uv_layout_image[..., 0][
        uv_layout_image[..., 3] >= 1
    ]

    canny[canny < 64] = 0  # noqa: PLR2004
    canny[canny >= 192] = 255  # noqa: PLR2004
    canny = 255 - canny

    # Stack the canny result into 3 channels (RGB)
    canny = np.stack((canny, canny, canny), axis=-1)
    canny = canny.astype(np.uint8)

    # create the pipe

    texture_input = texture_input[..., :3]

    # create the base of the final texture
    texture_input = cv2.resize(
        texture_input,
        (
            int(process_parameter.texture_resolution),
            int(process_parameter.texture_resolution),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    pipe = create_diffusion_pipeline(process_parameter)

    output_image = run_pipeline(
        pipe=pipe,
        process_parameter=process_parameter,
        input_img=texture_input,
        uv_mask=mask,
        canny_img=canny,
        normal_img=baked_texture_dict["normal"],
        depth_img=np.zeros_like(canny),
        progress_callback=progress_callback,
        strength=process_parameter.denoise_strength,
        guidance_scale=process_parameter.guidance_scale,
        num_inference_steps=process_parameter.num_inference_steps,
    )

    # TODO: Use the position bake information to mean over xyz distance

    # remove alpha
    output_image = np.array(output_image)[..., :3]

    cv2.imwrite(
        str(process_parameter.output_path / "texture_input_uv_pass.png"),
        cv2.cvtColor(texture_input, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(process_parameter.output_path / "mask_uv_pass.png"),
        cv2.cvtColor(mask, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(process_parameter.output_path / "canny_uv_pass.png"),
        cv2.cvtColor(canny, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(process_parameter.output_path / "output_image_uv_pass.png"),
        cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR),
    )

    return output_image
