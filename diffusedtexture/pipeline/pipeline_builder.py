from pathlib import Path

import bpy
import torch
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
    StableDiffusionXLControlNetUnionInpaintPipeline,
)

from .controlnet_config import build_controlnet_config


def create_diffusion_pipeline(
    context: bpy.context,
) -> (
    StableDiffusionControlNetInpaintPipeline
    | StableDiffusionXLControlNetUnionInpaintPipeline
):
    from diffusers import (
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionXLControlNetUnionInpaintPipeline,
    )

    config = build_controlnet_config(context.scene)
    pipe_cls = None

    if context.scene.sd_version == "sd15":
        pipe_cls = StableDiffusionControlNetInpaintPipeline
    elif context.scene.sd_version == "sdxl":
        pipe_cls = StableDiffusionXLControlNetUnionInpaintPipeline
    else:
        msg = "Unknown SD version: must be 'sd15' or 'sdxl'"
        raise ValueError(msg)

    controlnets = config[context.scene.mesh_complexity]["controlnets"]

    if str(context.scene.checkpoint_path).endswith(".safetensors"):
        pipe = pipe_cls.from_single_file(
            context.scene.checkpoint_path,
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        )
    elif str(context.scene.checkpoint_path).endswith((".ckpt", ".pt", ".pth", ".bin")):
        pipe = pipe_cls.from_single_file(
            context.scene.checkpoint_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        )
    else:
        pipe = pipe_cls.from_pretrained(
            context.scene.checkpoint_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
        )

    # LoRA
    if context.scene.num_loras > 0:
        for lora in context.scene.lora_models:
            pipe.load_lora_weights(
                str(Path(lora.path).parent),
                weight_name=str(Path(lora.path).name),
            )
            pipe.fuse_lora(lora_scale=lora.strength)

    # IPAdapter
    if context.scene.use_ipadapter:
        if context.scene.sd_version == "sd15":
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
        else:
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
        pipe.set_ip_adapter_scale(context.scene.ipadapter_strength)

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe
