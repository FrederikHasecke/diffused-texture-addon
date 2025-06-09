from enum import Enum


class NumCameras(Enum):
    """Enum of available cameras."""

    four = 4
    nine = 9
    sixteen = 16


class stable_diffusion_paths(Enum):
    """Default Setting for SD paths."""

    sd15_ckpt = "runwayml/stable-diffusion-v1-5"
    sd15_cn_canny = "lllyasviel/sd-controlnet-canny"
    sd15_cn_normal = "lllyasviel/sd-controlnet-normal"
    sd15_cn_depth = "lllyasviel/sd-controlnet-depth"
    sdxl_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_cn_union = "xinsir/controlnet-union-sdxl-1.0"
