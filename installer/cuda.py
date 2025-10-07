import os
import re

from .paths import run

# Exposed enum for Blender UI
CUDA_ENUM_ITEMS = [
    ("AUTO", "Automatically detect", "Detect NVIDIA CUDA; fallback to CPU"),
    ("cpu", "CPU only", "Install CPU-only PyTorch build"),
    ("cu126", "CUDA 12.6", "PyTorch wheels for CUDA 12.6"),
    ("cu128", "CUDA 12.8", "PyTorch wheels for CUDA 12.8"),
    ("cu129", "CUDA 12.9", "PyTorch wheels for CUDA 12.9"),
    ("rocm6.3", "ROCm 6.3 (AMD)", "PyTorch wheels for ROCm 6.3 (Linux/AMD)"),
]

SUPPORTED_CHANNELS = {"cpu", "cu126", "cu128", "cu129", "rocm6.3"}


def torch_index_url(channel: str) -> tuple[str, str]:
    if channel == "cpu":
        return ("https://download.pytorch.org/whl/cpu", "PyTorch CPU")
    if channel == "rocm6.3":
        return ("https://download.pytorch.org/whl/rocm6.3", "PyTorch ROCm 6.3")
    # CUDA channels
    return (f"https://download.pytorch.org/whl/{channel}", f"PyTorch {channel.upper()}")


def _parse_cuda_from_nvidia_smi(text: str) -> tuple[int, int] | None:
    m = re.search(r"CUDA\s+Version:\s*([0-9]+)\.([0-9]+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _parse_cuda_from_nvcc(text: str) -> tuple[int, int] | None:
    m = re.search(r"release\s+([0-9]+)\.([0-9]+)", text, flags=re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _detect_cuda_version() -> tuple[int, int] | None:
    rc, out = run(["nvidia-smi"])
    if rc == 0:
        ver = _parse_cuda_from_nvidia_smi(out or "")
        if ver:
            return ver
    rc, out = run(["nvcc", "--version"])
    if rc == 0:
        ver = _parse_cuda_from_nvcc(out or "")
        if ver:
            return ver
    # Environment hint (CUDA_PATH / CUDA_PATH_V12_9 / etc.)
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH") and v:
            m = re.search(r"(\d+)[._](\d+)", k) or re.search(r"(\d+)[._](\d+)", v)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def _map_cuda_to_channel(ver: tuple[int, int]) -> str:
    major, minor = ver
    if (major, minor) >= (12, 9):
        return "cu129"
    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    # Older -> default to CPU to avoid mismatched wheels
    return "cpu"


def normalize_choice(choice: str) -> str:
    c = choice.lower()
    if c == "auto":
        ver = _detect_cuda_version()
        return _map_cuda_to_channel(ver) if ver else "cpu"
    # user-picked explicit channels (including rocm6.3 / cpu)
    return c if c in SUPPORTED_CHANNELS else "cpu"
