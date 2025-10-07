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


def _pick_dtype(device: str):  # noqa: ANN202
    import torch

    # Use fp32 on CPU to avoid slow/failing fp16 kernels; use bf16 on modern GPUs
    if device in {"cpu", "mps"}:
        return torch.float32
    # cuda/rocm
    try:
        major, _ = torch.cuda.get_device_capability(0)
        if (
            major >= 8  # noqa: PLR2004
        ):  # Ampere+ generally supports bf16 reasonably well
            return torch.bfloat16
    except Exception:  # noqa: BLE001, S110
        pass
    return torch.float16


def create_diffusion_pipeline(  # noqa: C901, PLR0912
    process_parameter: ProcessParameter,
) -> Any:  # noqa: ANN401
    try:
        import torch  # noqa: F401
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

    device = _pick_device()

    common_kwargs = {
        "safety_checker": None,
        "requires_safety_checker": False,
    }

    if ckpt.endswith(".safetensors"):
        pipe = pipe_cls.from_single_file(
            pretrained_model_link_or_path=ckpt,
            controlnet=controlnets,
            use_safetensors=True,
            **common_kwargs,
        )
    elif ckpt.endswith((".ckpt", ".pt", ".pth", ".bin")):
        pipe = pipe_cls.from_single_file(
            pretrained_model_link_or_path=ckpt,
            controlnet=controlnets,
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

    if device == "cpu":
        pipe.to("cpu")
    elif device in ("cuda", "mps"):
        pipe.to(device)
        pipe.enable_model_cpu_offload()
    return pipe
