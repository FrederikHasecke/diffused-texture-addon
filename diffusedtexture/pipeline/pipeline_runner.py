import torch
from PIL import Image
from ...utils import image_to_numpy
from .controlnet_config import build_controlnet_config


def run_pipeline(
    pipe,
    scene,
    input_image,
    uv_mask,
    canny_img,
    normal_img,
    depth_img,
    strength=1.0,
    guidance_scale=7.5,
    num_inference_steps=None,
    denoising_start=None,
    denoising_end=None,
    output_type="pil",
):
    config = build_controlnet_config(scene)
    complexity = scene.mesh_complexity

    control_images = []
    for entry in config[complexity]["inputs"]:
        image_map = {"depth": depth_img, "canny": canny_img, "normal": normal_img}
        control_images.append(Image.fromarray(image_map[entry]))

    if num_inference_steps is None:
        num_inference_steps = scene.num_inference_steps

    try:
        kwargs = {
            "prompt": scene.my_prompt,
            "negative_prompt": scene.my_negative_prompt,
            "image": input_image,
            "mask_image": Image.fromarray(uv_mask),
            "control_image": control_images,
            "ip_adapter_image": image_to_numpy(scene.ipadapter_image)
            if scene.use_ipadapter
            else None,
            "num_images_per_prompt": 1,
            "controlnet_conditioning_scale": config[complexity]["conditioning_scale"],
            "num_inference_steps": num_inference_steps,
            "denoising_start": denoising_start,
            "denoising_end": denoising_end,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "output_type": output_type,
        }

        if scene.controlnet_type == "UNION":
            kwargs["control_mode"] = config[complexity]["control_mode"]

        return pipe(**kwargs).images

    except torch.cuda.OutOfMemoryError:
        del pipe
        torch.cuda.empty_cache()
        return None
