import torch
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
    StableDiffusionXLControlNetUnionInpaintPipeline,
)
from PIL import Image

from ...blender_operations import ProcessParameter
from ...utils import image_to_numpy
from .controlnet_config import build_controlnet_config


def run_pipeline(  # noqa: PLR0913
    pipe: StableDiffusionControlNetInpaintPipeline
    | StableDiffusionXLControlNetUnionInpaintPipeline,
    process_parameter: ProcessParameter,
    input_img: Image,
    uv_mask: Image,
    canny_img: Image,
    normal_img: Image,
    depth_img: Image,
    progress_callback: callable,
    strength: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> Image:
    config = build_controlnet_config(process_parameter)
    complexity = process_parameter.mesh_complexity

    control_images = []
    for entry in config[complexity]["inputs"]:
        image_map = {"depth": depth_img, "canny": canny_img, "normal": normal_img}
        control_images.append(image_map[entry])

    try:
        kwargs = {
            "prompt": process_parameter.my_prompt,
            "negative_prompt": process_parameter.my_negative_prompt,
            "image": input_img,
            "mask_image": uv_mask,
            "control_image": control_images,
            "ip_adapter_image": Image.fromarray(
                image_to_numpy(process_parameter.ipadapter_image),
            )
            if process_parameter.use_ipadapter
            else None,
            "num_images_per_prompt": 1,
            "controlnet_conditioning_scale": config[complexity]["conditioning_scale"],
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
        }

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

        return pipe(**kwargs).images[0]

    except torch.cuda.OutOfMemoryError:
        del pipe
        torch.cuda.empty_cache()
        return None
