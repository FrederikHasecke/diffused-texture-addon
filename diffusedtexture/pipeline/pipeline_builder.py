from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionXLControlNetUnionInpaintPipeline,
        )
    except ModuleNotFoundError:
        StableDiffusionControlNetInpaintPipeline = None  # type: ignore[assignment]
        StableDiffusionXLControlNetUnionInpaintPipeline = None  # type: ignore[assignment]


from ...blender_operations import ProcessParameter
from ...model_support import require_supported_sd_version
from .controlnet_config import build_controlnet_config


def _pick_device() -> str:
    try:
        import torch  # local import

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"  # noqa: TRY300
    except Exception:  # noqa: BLE001
        return "cpu"


def create_diffusion_pipeline(  # noqa: C901, PLR0912, PLR0915
    process_parameter: ProcessParameter,
) -> Any:  # noqa: ANN401
    try:
        import torch
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionXLControlNetUnionInpaintPipeline,
        )

    except ModuleNotFoundError:
        return None

    if (
        StableDiffusionControlNetInpaintPipeline is None
        or StableDiffusionXLControlNetUnionInpaintPipeline is None
    ):
        return None

    model_version = require_supported_sd_version(process_parameter.sd_version)
    config = build_controlnet_config(process_parameter)
    pipe_cls = None

    if model_version == "sd15":
        pipe_cls = StableDiffusionControlNetInpaintPipeline
        dtype = process_parameter.dtype or "float16"
    elif model_version == "sdxl":
        pipe_cls = StableDiffusionXLControlNetUnionInpaintPipeline
        dtype = process_parameter.dtype or "float16"
    else:
        msg = f"Unexpected supported model: {model_version}"
        raise RuntimeError(msg)

    controlnets = config[process_parameter.mesh_complexity]["controlnets"]

    # Build pipeline from checkpoint or Hub
    ckpt = str(process_parameter.checkpoint_path)

    device = _pick_device()
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float16)

    common_kwargs = {
        "safety_checker": None,
        "requires_safety_checker": False,
    }

    if ckpt.endswith(".safetensors"):
        pipe = pipe_cls.from_single_file(
            pretrained_model_link_or_path=ckpt,
            controlnet=controlnets,
            use_safetensors=True,
            torch_dtype=torch_dtype,
            **common_kwargs,
        )
    elif ckpt.endswith((".ckpt", ".pt", ".pth", ".bin")):
        pipe = pipe_cls.from_single_file(
            pretrained_model_link_or_path=ckpt,
            controlnet=controlnets,
            torch_dtype=torch_dtype,
            **common_kwargs,
        )
    else:
        pipe = pipe_cls.from_pretrained(
            ckpt,
            controlnet=controlnets,
            use_safetensors=True,
            torch_dtype=torch_dtype,
            **common_kwargs,
        )

    # LoRA
    if process_parameter.num_loras > 0:
        for lora in process_parameter.lora_models:
            pipe.load_lora_weights(
                str(Path(lora["path"]).parent),
                weight_name=str(Path(lora["path"]).name),
            )
            pipe.fuse_lora(lora_scale=lora["strength"])

    # IPAdapter
    if process_parameter.use_ipadapter:
        if model_version == "sd15":
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
        elif model_version == "sdxl":
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
        pipe.set_ip_adapter_scale(process_parameter.ipadapter_strength)

    if device == "cpu":
        pipe.to("cpu")
    else:
        pipe.to(device)
        pipe.enable_model_cpu_offload()
    return pipe
