import os

from PIL import Image

from ..utils import image_to_numpy

# import torch
# from diffusers import (
#     StableDiffusionControlNetInpaintPipeline,
#     ControlNetModel,
# )
# from transformers import CLIPVisionModelWithProjection


def get_controlnet_config(scene):
    import torch
    from diffusers import (
        ControlNetModel,
    )

    # Create a dictionary to map model complexities to their corresponding controlnet weights and inputs
    controlnet_config = {
        "LOW": {
            "conditioning_scale": [scene.depth_controlnet_strength],  # 0.8,
            "controlnets": [
                ControlNetModel.from_pretrained(
                    scene.depth_controlnet_path, torch_dtype=torch.float16
                ),
            ],
            "inputs": ["depth"],
        },
        "MEDIUM": {
            "conditioning_scale": [
                scene.depth_controlnet_strength,
                scene.canny_controlnet_strength,
            ],  # [0.7, 0.8],
            "controlnets": [
                ControlNetModel.from_pretrained(
                    scene.depth_controlnet_path, torch_dtype=torch.float16
                ),
                ControlNetModel.from_pretrained(
                    scene.canny_controlnet_path, torch_dtype=torch.float16
                ),
            ],
            "inputs": ["depth", "canny"],
        },
        "HIGH": {
            "conditioning_scale": [
                scene.depth_controlnet_strength,
                scene.canny_controlnet_strength,
                scene.normal_controlnet_strength,
            ],  # [1.0, 0.9, 0.9],
            "controlnets": [
                ControlNetModel.from_pretrained(
                    scene.depth_controlnet_path, torch_dtype=torch.float16
                ),
                ControlNetModel.from_pretrained(
                    scene.canny_controlnet_path, torch_dtype=torch.float16
                ),
                ControlNetModel.from_pretrained(
                    scene.normal_controlnet_path, torch_dtype=torch.float16
                ),
            ],
            "inputs": ["depth", "canny", "normal"],
        },
    }
    return controlnet_config


def create_pipeline(scene):

    # re-import if hf_home was re-set
    import torch
    from diffusers import (
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
    )

    controlnet_config = get_controlnet_config(scene)

    if scene.sd_version == "sd15":

        # Load the model from a checkpoint if provided as safe tensor or checkpoint
        if str(scene.checkpoint_path).endswith(".safetensors"):
            pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                scene.checkpoint_path,
                use_safetensors=True,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
            )

        # check if the checkpoint_path ends with .ckpt, .pt, .pth, .bin or any other extension, then load the model from the checkpoint
        elif str(scene.checkpoint_path).endswith((".ckpt", ".pt", ".pth", ".bin")):
            try:
                pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                    scene.checkpoint_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                )
            except:
                # untested so raise verbose error
                raise ValueError(
                    "Invalid checkpoint path provided. Please provide a valid checkpoint path"
                )

        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                scene.checkpoint_path,
                controlnet=controlnet_config[scene.mesh_complexity]["controlnets"],
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
            )

    elif scene.sd_version == "sdxl":

        if str(scene.checkpoint_path).endswith(".safetensors"):
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                scene.checkpoint_path,
                use_safetensors=True,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
            )

        elif str(scene.checkpoint_path).endswith((".ckpt", ".pt", ".pth", ".bin")):
            try:
                pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                    scene.checkpoint_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                )
            except:
                # untested so raise verbose error
                raise ValueError(
                    "Invalid checkpoint path provided. Please provide a valid checkpoint path"
                )

        else:
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                scene.checkpoint_path,
                controlnet=controlnet_config[scene.mesh_complexity]["controlnets"],
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
            )
    else:
        raise ValueError("Invalid SD Version, can only be 'sd15' or 'sdxl'")

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
        if scene.sd_version == "sd15":
            pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
            )
        elif scene.sd_version == "sdxl":
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
        else:
            raise ValueError("Invalid SD Version, can only be 'sd15' or 'sdxl'")

        pipe.set_ip_adapter_scale(scene.ipadapter_strength)

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    return pipe


def infer_pipeline(
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

    controlnet_config = get_controlnet_config(scene)

    # run the pipeline
    control_images = []

    for entry in controlnet_config[scene.mesh_complexity]["inputs"]:
        if entry == "depth":
            control_images.append(Image.fromarray(depth_img))
        elif entry == "canny":
            control_images.append(Image.fromarray(canny_img))
        elif entry == "normal":
            control_images.append(Image.fromarray(normal_img))

    ip_adapter_image = None
    if scene.use_ipadapter:
        ip_adapter_image = image_to_numpy(scene.ipadapter_image)

    if num_inference_steps is None:
        num_inference_steps = scene.num_inference_steps

    output = pipe(
        prompt=scene.my_prompt,
        negative_prompt=scene.my_negative_prompt,
        image=input_image,
        mask_image=Image.fromarray(uv_mask),
        control_image=control_images,
        ip_adapter_image=ip_adapter_image,
        num_images_per_prompt=1,
        controlnet_conditioning_scale=controlnet_config[scene.mesh_complexity][
            "conditioning_scale"
        ],
        num_inference_steps=num_inference_steps,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        strength=strength,
        guidance_scale=guidance_scale,
        output_type=output_type,
    ).images

    return output
