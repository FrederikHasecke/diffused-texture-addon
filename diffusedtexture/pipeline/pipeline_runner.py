from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
        StableDiffusionControlNetInpaintPipeline,
    )
    from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
        StableDiffusionXLControlNetUnionInpaintPipeline,
    )
    from PIL import Image

    PipeType = (
        StableDiffusionControlNetInpaintPipeline
        | StableDiffusionXLControlNetUnionInpaintPipeline
    )
    from ...blender_operations import ProcessParameter

else:
    Image = object  # runtime placeholder
    PipeType = object

from ...utils import image_to_numpy
from .controlnet_config import build_controlnet_config


def run_pipeline(  # noqa: PLR0913
    pipe: PipeType,
    process_parameter: ProcessParameter,
    input_img: Image,
    uv_mask: Image,
    canny_img: Image,
    normal_img: Image,
    depth_img: Image,
    progress_callback: Callable,
    strength: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> Image:
    # Lazy imports so the add-on can register without deps.
    try:
        import torch
        from PIL import Image
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
                Image.fromarray(image_to_numpy(process_parameter.ipadapter_image))
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
            pipe: PipeType,
            step_index: int,
            timestep: int,  # noqa: ARG001
            callback_kwargs: dict,
        ) -> dict:
            progress = step_index / pipe.num_timesteps
            percent = int(100 * progress)
            progress_callback(percent)
            return callback_kwargs if callback_kwargs is not None else {}

        kwargs["callback_on_step_end"] = pipe_progress_callback

        return pipe(**kwargs).images[0]

    except torch.cuda.OutOfMemoryError:
        del pipe
        torch.cuda.empty_cache()
        return None
