import os

from PIL import Image

from ..utils import image_to_numpy

# import torch
# from diffusers import (
#     StableDiffusionControlNetInpaintPipeline,
#     ControlNetModel,
# )
# from transformers import CLIPVisionModelWithProjection


def get_controlnet_config():
    import torch
    from diffusers import (
        ControlNetModel,
    )

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
            "conditioning_scale": [0.7, 0.8],
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
            "conditioning_scale": [1.0, 0.9, 0.9],
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
    return controlnet_config


def create_first_pass_pipeline(scene):

    # re-import if hf_home was re-set
    import torch
    from diffusers import StableDiffusionControlNetInpaintPipeline

    controlnet_config = get_controlnet_config()

    # TODO: Add the options for LoRA and IPAdapter
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        # "runwayml/stable-diffusion-inpainting",
        "runwayml/stable-diffusion-v1-5",
        # "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        controlnet=controlnet_config[scene.mesh_complexity]["controlnets"],
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
    )

    if scene.num_loras > 0:

        for lora in scene.lora_models:

            # Extract the directory (everything but the file name)
            file_path = os.path.dirname(lora.path)

            # Extract the file name (just the file name, including extension)
            file_name = os.path.basename(lora.path)

            # Load the LoRA weights using the extracted path and file name
            pipe.load_lora_weights(file_path, weight_name=file_name)

            pipe.fuse_lora(lora_scale=lora.strength)

    if scene.use_ipadapter:
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
        )

        pipe.set_ip_adapter_scale(scene.ipadapter_strength)

    pipe.enable_model_cpu_offload()

    return pipe


def infer_first_pass_pipeline(
    pipe,
    scene,
    input_image,
    uv_mask,
    canny_img,
    normal_img,
    depth_img,
    strength=1.0,
    guidance_scale=7.5,
):

    controlnet_config = get_controlnet_config()

    # run the pipeline
    control_images = []

    for entry in controlnet_config[scene.mesh_complexity]["inputs"]:
        if entry == "depth":
            control_images.append(
                Image.fromarray(
                    depth_img
                )  # .resize((512, 512), Image.Resampling.LANCZOS)
            )
        elif entry == "canny":
            control_images.append(
                Image.fromarray(
                    canny_img
                )  # .resize((512, 512), Image.Resampling.LANCZOS)
            )
        elif entry == "normal":
            control_images.append(
                Image.fromarray(
                    normal_img
                )  # .resize((512, 512), Image.Resampling.LANCZOS)
            )

    ip_adapter_image = None
    if scene.use_ipadapter:
        ip_adapter_image = image_to_numpy(scene.ipadapter_image)

    output = (
        pipe(
            prompt=scene.my_prompt,
            negative_prompt="reflective, shiny, shadow, " + scene.my_negative_prompt,
            image=Image.fromarray(input_image),
            mask_image=Image.fromarray(uv_mask),
            control_image=control_images,
            ip_adapter_image=ip_adapter_image,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=controlnet_config[scene.mesh_complexity][
                "conditioning_scale"
            ],
            num_inference_steps=50,
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        # .resize((canny_img.shape[0], canny_img.shape[1]), Image.Resampling.LANCZOS)
    )

    return output


# def create_uv_pass_pipeline(scene):

#     # TODO: Add the options for LoRA and IPAdapter
#     pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         controlnet=[
#             ControlNetModel.from_pretrained(
#                 "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
#             )
#         ],
#         torch_dtype=torch.float16,
#         use_safetensors=True,
#         safety_checker=None,
#     )

#     if scene.num_loras > 0:

#         for lora in scene.lora_models:

#             # Extract the directory (everything but the file name)
#             file_path = os.path.dirname(lora.path)

#             # Extract the file name (just the file name, including extension)
#             file_name = os.path.basename(lora.path)

#             # Load the LoRA weights using the extracted path and file name
#             pipe.load_lora_weights(file_path, weight_name=file_name)

#             pipe.fuse_lora(lora_scale=lora.strength)

#     if scene.use_ipadapter:
#         pipe.load_ip_adapter(
#             "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
#         )

#         pipe.set_ip_adapter_scale(scene.ipadapter_strength)

#     pipe.enable_model_cpu_offload()

#     return pipe


# def infer_uv_pass_pipeline(pipe, scene, input_image, uv_mask, canny_img, strength=1.0):
#     # run the pipeline
#     control_images = []

#     control_images.append(
#         Image.fromarray(canny_img)  # .resize((512, 512), Image.Resampling.LANCZOS)
#     )

#     ip_adapter_image = None
#     if scene.use_ipadapter:
#         ip_adapter_image = image_to_numpy(scene.ipadapter_image)

#     output = (
#         pipe(
#             prompt="A flat image of a texture of " + scene.my_prompt,
#             negative_prompt=scene.my_negative_prompt,
#             image=Image.fromarray(input_image),
#             mask_image=Image.fromarray(uv_mask),
#             control_image=control_images,
#             ip_adapter_image=ip_adapter_image,
#             num_images_per_prompt=1,
#             controlnet_conditioning_scale=[0.2],
#             num_inference_steps=50,
#             strength=strength,
#             # guidance_scale=10.0,
#         ).images[0]
#         # .resize((canny_img.shape[0], canny_img.shape[1]), Image.Resampling.LANCZOS)
#     )

#     return output
