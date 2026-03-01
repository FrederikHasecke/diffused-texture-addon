from collections.abc import Callable

import numpy as np
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLControlNetUnionInpaintPipeline,
)
from numpy.typing import NDArray
from PIL import Image

from ...blender_operations import ProcessParameter
from .controlnet_config import build_controlnet_config


def _to_pil_image(img: NDArray | Image.Image | None) -> Image.Image | None:
    if img is None:
        return None

    if isinstance(img, Image.Image):
        return img

    arr = np.asarray(img)
    if arr.ndim == 2:  # noqa: PLR2004
        arr = np.repeat(arr[..., None], 3, axis=2)

    if arr.ndim != 3:  # noqa: PLR2004
        msg = "IPAdapter image must be a 2D or 3D array."
        raise ValueError(msg)

    if arr.shape[2] == 4:  # noqa: PLR2004
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        max_value = float(np.max(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(arr)


def run_pipeline(  # noqa: PLR0913
    pipe: StableDiffusionControlNetInpaintPipeline
    | StableDiffusionXLControlNetUnionInpaintPipeline,
    process_parameter: ProcessParameter,
    input_img: Image.Image,
    uv_mask: Image.Image,
    canny_img: Image.Image,
    normal_img: Image.Image,
    depth_img: Image.Image,
    progress_callback: Callable,
    strength: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> Image.Image | None:
    # Lazy imports so the add-on can register without deps.
    try:
        import torch
    except ModuleNotFoundError:
        return None

    config = build_controlnet_config(process_parameter)
    complexity = process_parameter.mesh_complexity

    image_map = {"depth": depth_img, "canny": canny_img, "normal": normal_img}
    control_images = [image_map[key] for key in config[complexity]["inputs"]]

    if process_parameter.sd_version == "sdxl":
        control_mode = []
        if "depth" in config[complexity]["inputs"]:
            control_mode.append(1)
        if "canny" in config[complexity]["inputs"]:
            control_mode.append(3)
        if "normal" in config[complexity]["inputs"]:
            control_mode.append(4)

    try:
        kwargs = {
            "prompt": process_parameter.my_prompt,
            "negative_prompt": process_parameter.my_negative_prompt,
            "image": input_img,
            "mask_image": uv_mask,
            "control_image": control_images,
            "ip_adapter_image": (
                _to_pil_image(process_parameter.ipadapter_image)
                if process_parameter.use_ipadapter
                else None
            ),
            "num_images_per_prompt": 1,
            "controlnet_conditioning_scale": config[complexity]["conditioning_scale"],
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
        }

        if process_parameter.sd_version == "sdxl":
            kwargs["control_mode"] = control_mode

        def pipe_progress_callback(
            pipe: StableDiffusionControlNetInpaintPipeline
            | StableDiffusionXLControlNetUnionInpaintPipeline,
            step_index: int,
            timestep: int,  # noqa: ARG001
            callback_kwargs: dict,
        ) -> dict:
            progress = step_index / pipe.num_timesteps
            percent = int(100 * progress)
            progress_callback(percent)
            return callback_kwargs if callback_kwargs is not None else {}

        kwargs["callback_on_step_end"] = pipe_progress_callback

        return pipe.__call__(**kwargs).images[0]

    except torch.cuda.OutOfMemoryError:
        del pipe
        torch.cuda.empty_cache()
        return None
