from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
        StableDiffusionControlNetInpaintPipeline,
    )
    from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
        StableDiffusionXLControlNetUnionInpaintPipeline,
    )

    from ...blender_operations import ProcessParameter


from .controlnet_config import build_controlnet_config


def _pick_device() -> str:
    try:
        import torch  # local import

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001, S110
        pass
    return "cpu"


def create_diffusion_pipeline(  # noqa: C901, PLR0912
    process_parameter: ProcessParameter,
) -> (
    StableDiffusionControlNetInpaintPipeline
    | StableDiffusionXLControlNetUnionInpaintPipeline
    | None
):
    try:
        import torch
        from diffusers import (  # noqa: PLC0415, RUF100
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionXLControlNetUnionInpaintPipeline,
        )
    except ModuleNotFoundError:
        return None

    config = build_controlnet_config(process_parameter)
    pipe_cls = None

    if process_parameter.sd_version == "sd15":
        pipe_cls = StableDiffusionControlNetInpaintPipeline
    elif process_parameter.sd_version == "sdxl":
        pipe_cls = StableDiffusionXLControlNetUnionInpaintPipeline
    else:
        msg = "Unknown SD version: must be 'sd15' or 'sdxl'"
        raise ValueError(msg)

    controlnets = config[process_parameter.mesh_complexity]["controlnets"]

    # Build pipeline from checkpoint or Hub
    ckpt = str(process_parameter.checkpoint_path)
    common_kwargs = {"torch_dtype": torch.float16, "safety_checker": None}

    if ckpt.endswith(".safetensors"):
        pipe = pipe_cls.from_single_file(
            pretrained_model_link_or_path=ckpt,
            controlnet=controlnets,
            use_safetensors=True,
            variant="fp16",
            **common_kwargs,
        )
    elif ckpt.endswith((".ckpt", ".pt", ".pth", ".bin")):
        pipe = pipe_cls.from_single_file(
            pretrained_model_link_or_path=ckpt,
            controlnet=controlnets,
            variant="fp16",
            **common_kwargs,
        )
    else:
        pipe = pipe_cls.from_pretrained(
            ckpt,
            controlnet=controlnets,
            use_safetensors=True,
            **common_kwargs,
        )

    # LoRA
    if process_parameter.num_loras > 0:
        for lora in process_parameter.lora_models:
            pipe.load_lora_weights(
                str(Path(lora.path).parent),
                weight_name=str(Path(lora.path).name),
            )
            pipe.fuse_lora(lora_scale=lora.strength)

    # IPAdapter
    if process_parameter.use_ipadapter:
        if process_parameter.sd_version == "sd15":
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
        pipe.set_ip_adapter_scale(process_parameter.ipadapter_strength)

    device = _pick_device()
    pipe.to(device)
    if device != "cpu":
        # Only useful when there is a GPU backend
        pipe.enable_model_cpu_offload()
    return pipe
