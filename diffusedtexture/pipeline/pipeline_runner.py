import torch
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
    StableDiffusionXLControlNetUnionInpaintPipeline,
)
from PIL import Image

from ...blender_operations import ProcessParameters
from ...utils import image_to_numpy
from .controlnet_config import build_controlnet_config


def run_pipeline(  # noqa: PLR0913
    pipe: StableDiffusionControlNetInpaintPipeline
    | StableDiffusionXLControlNetUnionInpaintPipeline,
    process_parameters: ProcessParameters,
    input_image: Image,
    uv_mask: Image,
    canny_img: Image,
    normal_img: Image,
    depth_img: Image,
    strength: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int | None = None,
) -> Image:
    config = build_controlnet_config(process_parameters)
    complexity = process_parameters.mesh_complexity

    control_images = []
    for entry in config[complexity]["inputs"]:
        image_map = {"depth": depth_img, "canny": canny_img, "normal": normal_img}
        control_images.append(image_map[entry])

    if num_inference_steps is None:
        num_inference_steps = process_parameters.num_inference_steps

    try:
        kwargs = {
            "prompt": process_parameters.my_prompt,
            "negative_prompt": process_parameters.my_negative_prompt,
            "image": input_image,
            "mask_image": uv_mask,
            "control_image": control_images,
            "ip_adapter_image": Image.fromarray(
                image_to_numpy(process_parameters.ipadapter_image),
            )
            if process_parameters.use_ipadapter
            else None,
            "num_images_per_prompt": 1,
            "controlnet_conditioning_scale": config[complexity]["conditioning_scale"],
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
        }

        return pipe(**kwargs).images[0]

    except torch.cuda.OutOfMemoryError:
        del pipe
        torch.cuda.empty_cache()
        return None
