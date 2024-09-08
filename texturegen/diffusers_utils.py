import os

os.environ["HF_HOME"] = r"G:\Huggingface_cache"

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from PIL import Image


# Create a dictionary to map model complexities to their corresponding controlnet weights and inputs
controlnet_config = {
    "LOW": {
        "conditioning_scale": [0.8],
        "controlnets": [
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            ),
        ],
        "inputs": ["depth"],
    },
    "MEDIUM": {
        "conditioning_scale": [0.8, 0.75],
        "controlnets": [
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
            ),
        ],
        "inputs": ["depth", "canny"],
    },
    "HIGH": {
        "conditioning_scale": [1.0, 1.0, 1.0],
        "controlnets": [
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16
            ),
        ],
        "inputs": ["depth", "canny", "normal"],
    },
}


def create_first_pass_pipeline(scene):

    # TODO: Add the options for LoRA and IPAdapter
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        # "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        controlnet=controlnet_config[scene.mesh_complexity]["controlnets"],
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
    )

    pipe.enable_model_cpu_offload()

    return pipe


def infer_first_pass_pipeline(pipe, scene, canny_img, normal_img, depth_img):
    # run the pipeline
    images = []

    for entry in controlnet_config[scene.mesh_complexity]["inputs"]:
        if entry == "depth":
            images.append(
                Image.fromarray(depth_img).resize((512, 512), Image.Resampling.LANCZOS)
            )
        elif entry == "canny":
            images.append(
                Image.fromarray(canny_img).resize((512, 512), Image.Resampling.LANCZOS)
            )
        elif entry == "normal":
            images.append(
                Image.fromarray(normal_img).resize((512, 512), Image.Resampling.LANCZOS)
            )

    output = (
        pipe(
            prompt=scene.my_prompt,
            negative_prompt=scene.my_negative_prompt,
            image=images,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=controlnet_config[scene.mesh_complexity][
                "conditioning_scale"
            ],
            num_inference_steps=50,
            guidance_scale=10.0,
        )
        .images[0]
        .resize((canny_img.shape[0], canny_img.shape[1]), Image.Resampling.LANCZOS)
    )

    return output