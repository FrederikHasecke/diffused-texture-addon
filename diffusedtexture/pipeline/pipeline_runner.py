from collections.abc import Callable

import numpy as np
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLControlNetUnionInpaintPipeline,
)
from numpy.typing import NDArray
from PIL import Image

from ...blender_operations import ProcessParameter
from .controlnet_config import get_controlnet_static_config


class TextureGenerationCancelledError(RuntimeError):
    """Raised when the user cancels an in-flight generation."""

    MESSAGE = "Cancelled by user."


def _raise_generation_cancelled() -> None:
    msg = TextureGenerationCancelledError.MESSAGE
    raise TextureGenerationCancelledError(msg)


def _to_pil_image(img: NDArray | Image.Image | None) -> Image.Image | None:
    if img is None:
        return None

    if isinstance(img, Image.Image):
        return img

    arr = np.asarray(img)
    if arr.ndim == 2:  # noqa: PLR2004
        arr = np.repeat(arr[..., None], 3, axis=2)

    if arr.ndim != 3:  # noqa: PLR2004
        msg = "IPAdapter image must be a 2D or 3D array."
        raise ValueError(msg)

    if arr.shape[2] == 4:  # noqa: PLR2004
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        max_value = float(np.max(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(arr)


def run_pipeline(  # noqa: C901, PLR0913
    pipe: StableDiffusionControlNetInpaintPipeline
    | StableDiffusionXLControlNetUnionInpaintPipeline,
    process_parameter: ProcessParameter,
    input_img: Image.Image,
    uv_mask: Image.Image,
    canny_img: Image.Image,
    normal_img: Image.Image,
    depth_img: Image.Image,
    progress_callback: Callable,
    should_cancel: Callable[[], bool],
    strength: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> Image.Image:
    # Lazy imports so the add-on can register without deps.
    try:
        import torch
    except ModuleNotFoundError as exc:
        msg = (
            "Python dependencies missing. Open Preferences > Add-ons > DiffusedTexture "
            "> Install Python Dependencies, restart Blender, and try again."
        )
        raise RuntimeError(msg) from exc

    if pipe is None:
        msg = (
            "Diffusion pipeline is not initialized. Ensure dependencies are "
            "installed and "
            "the selected model and ControlNet paths are valid."
        )
        raise RuntimeError(msg)

    config = get_controlnet_static_config(process_parameter)

    image_map = {"depth": depth_img, "canny": canny_img, "normal": normal_img}
    control_images = [image_map[key] for key in config["inputs"]]

    try:
        kwargs = {
            "prompt": process_parameter.my_prompt,
            "negative_prompt": process_parameter.my_negative_prompt,
            "image": input_img,
            "mask_image": uv_mask,
            "control_image": control_images,
            "ip_adapter_image": (
                _to_pil_image(process_parameter.ipadapter_image)
                if process_parameter.use_ipadapter
                else None
            ),
            "num_images_per_prompt": 1,
            "controlnet_conditioning_scale": config["conditioning_scale"],
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
        }

        if process_parameter.sd_version == "sdxl":
            kwargs["control_mode"] = config["control_mode"]

        def pipe_progress_callback(
            pipe: StableDiffusionControlNetInpaintPipeline
            | StableDiffusionXLControlNetUnionInpaintPipeline,
            step_index: int,
            timestep: int,  # noqa: ARG001
            callback_kwargs: dict,
        ) -> dict:
            progress = step_index / pipe.num_timesteps
            percent = int(100 * progress)
            progress_callback(percent)
            if should_cancel():
                pipe._interrupt = True  # noqa: SLF001
            return callback_kwargs if callback_kwargs is not None else {}

        kwargs["callback_on_step_end"] = pipe_progress_callback

        if process_parameter.texture_seed > 0:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs["generator"] = torch.Generator(device).manual_seed(
                process_parameter.texture_seed,
            )

        output = pipe.__call__(**kwargs)
        if should_cancel() or getattr(pipe, "_interrupt", False):
            _raise_generation_cancelled()
        generated_images = getattr(output, "images", None)
        if generated_images is None or len(generated_images) == 0:
            msg = "Stable Diffusion inference produced no output images."
            raise RuntimeError(msg)  # noqa: TRY301
        return generated_images[0]

    except torch.cuda.OutOfMemoryError as exc:
        del pipe
        torch.cuda.empty_cache()
        msg = (
            "Stable Diffusion ran out of GPU memory (CUDA OOM). Reduce render/SD "
            "resolution, lower inference steps, or close other GPU-heavy apps and try "
            "again."
        )
        raise RuntimeError(msg) from exc
    except TextureGenerationCancelledError:
        raise
    except Exception as exc:
        msg = (
            "Stable Diffusion inference failed "
            f"({type(exc).__name__}: {exc!s}). Check model paths and generation "
            "settings, then try again."
        )
        raise RuntimeError(msg) from exc
