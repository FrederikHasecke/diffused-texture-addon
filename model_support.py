from typing import Any

SUPPORTED_MODEL_MESSAGE = "Supported models: sd15, sdxl."

MODEL_SUPPORT_MATRIX: dict[str, dict[str, Any]] = {
    "sd15": {
        "label": "Stable Diffusion 1.5",
        "supported": True,
        "reason": "",
        "default_dtype": "float16",
        "default_sd_resolution": 512,
        "supports_ipadapter": True,
        "checkpoint_path": "runwayml/stable-diffusion-v1-5",
        "controlnet_union_path": "",
        "depth_controlnet_path": "lllyasviel/sd-controlnet-depth",
        "canny_controlnet_path": "lllyasviel/sd-controlnet-canny",
        "normal_controlnet_path": "lllyasviel/sd-controlnet-normal",
    },
    "sdxl": {
        "label": "Stable Diffusion XL",
        "supported": True,
        "reason": "",
        "default_dtype": "float16",
        "default_sd_resolution": 1024,
        "supports_ipadapter": True,
        "checkpoint_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "controlnet_union_path": "xinsir/controlnet-union-sdxl-1.0",
        "depth_controlnet_path": "",
        "canny_controlnet_path": "",
        "normal_controlnet_path": "",
    },
    "flux": {
        "label": "Flux",
        "supported": False,
        "reason": "Flux support is not implemented in this add-on build.",
        "default_dtype": "bfloat16",
        "default_sd_resolution": 1024,
        "supports_ipadapter": False,
        "checkpoint_path": "black-forest-labs/FLUX.1-Depth-dev",
        "controlnet_union_path": "sayakpaul/FLUX.1-Depth-dev-nf4",
        "depth_controlnet_path": "",
        "canny_controlnet_path": "",
        "normal_controlnet_path": "",
    },
    "qwen": {
        "label": "Qwen",
        "supported": False,
        "reason": "Qwen support is not implemented in this add-on build.",
        "default_dtype": "bfloat16",
        "default_sd_resolution": 1328,
        "supports_ipadapter": False,
        "checkpoint_path": "Qwen/Qwen-Image",
        "controlnet_union_path": "InstantX/Qwen-Image-ControlNet-Union",
        "depth_controlnet_path": "",
        "canny_controlnet_path": "",
        "normal_controlnet_path": "",
    },
}

SUPPORTED_SD_VERSIONS = tuple(
    version for version, data in MODEL_SUPPORT_MATRIX.items() if data["supported"]
)
DEFAULT_SD_VERSION = SUPPORTED_SD_VERSIONS[0]


def _unsupported_model_message(sd_version: str | None) -> str:
    if sd_version in MODEL_SUPPORT_MATRIX:
        reason = MODEL_SUPPORT_MATRIX[sd_version]["reason"]
        if reason:
            return (
                f"Unsupported model '{sd_version}'. {SUPPORTED_MODEL_MESSAGE} {reason}"
            )
    return (
        f"Unsupported model '{sd_version}'. {SUPPORTED_MODEL_MESSAGE} "
        "Flux and Qwen are currently disabled."
    )


def require_supported_sd_version(sd_version: str | None) -> str:
    if sd_version in SUPPORTED_SD_VERSIONS:
        return str(sd_version)
    msg = _unsupported_model_message(sd_version)
    raise ValueError(msg)


def get_default_model_paths(sd_version: str) -> dict[str, str]:
    model_version = require_supported_sd_version(sd_version)
    data = MODEL_SUPPORT_MATRIX[model_version]
    return {
        "checkpoint_path": data["checkpoint_path"],
        "controlnet_union_path": data["controlnet_union_path"],
        "depth_controlnet_path": data["depth_controlnet_path"],
        "canny_controlnet_path": data["canny_controlnet_path"],
        "normal_controlnet_path": data["normal_controlnet_path"],
    }


def get_sd_version_enum_items() -> list[tuple[str, str, str]]:
    return [
        (
            version,
            str(MODEL_SUPPORT_MATRIX[version]["label"]),
            "Supported model",
        )
        for version in SUPPORTED_SD_VERSIONS
    ]


def get_default_sd_resolution(sd_version: str | None, custom_sd_resolution: int) -> int:
    if custom_sd_resolution:
        return int(custom_sd_resolution)
    model_version = require_supported_sd_version(sd_version)
    return int(MODEL_SUPPORT_MATRIX[model_version]["default_sd_resolution"])


def supports_ipadapter(sd_version: str | None) -> bool:
    if sd_version not in MODEL_SUPPORT_MATRIX:
        return False
    if not MODEL_SUPPORT_MATRIX[sd_version]["supported"]:
        return False
    return bool(MODEL_SUPPORT_MATRIX[sd_version]["supports_ipadapter"])


def unsupported_model_message(sd_version: str | None) -> str:
    return _unsupported_model_message(sd_version)
