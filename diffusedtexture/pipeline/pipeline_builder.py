import os
import torch
from .controlnet_config import build_controlnet_config


def create_diffusion_pipeline(scene):
    from diffusers import (
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetUnionInpaintPipeline,
    )

    config = build_controlnet_config(scene)
    pipe_cls = None

    if scene.sd_version == "sd15":
        pipe_cls = StableDiffusionControlNetInpaintPipeline
    elif scene.sd_version == "sdxl":
        if scene.controlnet_type == "UNION":
            pipe_cls = StableDiffusionXLControlNetUnionInpaintPipeline
        else:
            pipe_cls = StableDiffusionXLControlNetInpaintPipeline
    else:
        raise ValueError("Unknown SD version: must be 'sd15' or 'sdxl'")

    controlnets = config[scene.mesh_complexity]["controlnets"]

    if str(scene.checkpoint_path).endswith(".safetensors"):
        pipe = pipe_cls.from_single_file(
            scene.checkpoint_path,
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        )
    elif str(scene.checkpoint_path).endswith((".ckpt", ".pt", ".pth", ".bin")):
        pipe = pipe_cls.from_single_file(
            scene.checkpoint_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        )
    else:
        pipe = pipe_cls.from_pretrained(
            scene.checkpoint_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
        )

    # LoRA
    if scene.num_loras > 0:
        for lora in scene.lora_models:
            pipe.load_lora_weights(
                os.path.dirname(lora.path), weight_name=os.path.basename(lora.path)
            )
            pipe.fuse_lora(lora_scale=lora.strength)

    # IPAdapter
    if scene.use_ipadapter:
        if scene.sd_version == "sd15":
            pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
            )
        else:
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
        pipe.set_ip_adapter_scale(scene.ipadapter_strength)

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe
