import os
import re
import time

from .paths import run

# Exposed enum for Blender UI
CUDA_ENUM_ITEMS = [
    ("AUTO", "Automatically detect", "Detect NVIDIA CUDA; fallback to CPU"),
    ("cpu", "CPU only", "Install CPU-only PyTorch build"),
    ("cu126", "CUDA 12.6", "PyTorch wheels for CUDA 12.6"),
    ("cu128", "CUDA 12.8", "PyTorch wheels for CUDA 12.8"),
    ("cu129", "CUDA 12.9", "PyTorch wheels for CUDA 12.9"),
    ("cu130", "CUDA 13.0+", "PyTorch wheels for CUDA 13.0 and later"),
    ("rocm6.3", "ROCm 6.3 (AMD)", "PyTorch wheels for ROCm 6.3 (Linux/AMD)"),
]

SUPPORTED_CHANNELS = {"cpu", "cu126", "cu128", "cu129", "cu130", "rocm6.3"}
_DETECTION_TTL_SEC = 10.0
_detection_cache: tuple[float, tuple[int, int] | None] | None = None


def resolve_torch_install(
    channel: str,
    *,
    platform: str,
    blender_version: tuple[int, int, int],
) -> tuple[str, str, str]:
    """Return (channel, torch requirement, note) for installation.

    Blender on Windows up to 5.0 ships an older MSVC runtime. Newer PyTorch
    wheels (e.g. current cu130 builds) can fail to import with WinError 1114
    from c10.dll inside Blender's process. Pinning to 2.8.0 avoids this.
    """
    resolved_channel = channel
    if platform == "win32" and blender_version < (5, 1, 0):
        if resolved_channel == "cu130":
            resolved_channel = "cu129"
        if resolved_channel.startswith("rocm"):
            return (
                resolved_channel,
                "torch",
                ("ROCm wheels were requested on Windows and may not be available."),
            )
        build_tag = "cpu" if resolved_channel == "cpu" else resolved_channel
        note = (
            "Pinned PyTorch to 2.8.0 for Blender <= 5.0 on Windows to avoid "
            "c10.dll WinError 1114."
        )
        if channel == "cu130" and resolved_channel != channel:
            note = (
                f"{note} CUDA 13.0 wheels are unavailable for 2.8.0; "
                "using CUDA 12.9 wheels instead."
            )
        return resolved_channel, f"torch==2.8.0+{build_tag}", note
    return resolved_channel, "torch", ""


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


def detect_cuda_version() -> tuple[int, int] | None:
    global _detection_cache  # noqa: PLW0603
    now = time.time()
    if (
        _detection_cache is not None
        and (now - _detection_cache[0]) < _DETECTION_TTL_SEC
    ):
        return _detection_cache[1]

    rc, out = run(["nvidia-smi"])
    if rc == 0:
        ver = _parse_cuda_from_nvidia_smi(out or "")
        if ver:
            _detection_cache = (now, ver)
            return ver
    rc, out = run(["nvcc", "--version"])
    if rc == 0:
        ver = _parse_cuda_from_nvcc(out or "")
        if ver:
            _detection_cache = (now, ver)
            return ver
    # Environment hint (CUDA_PATH / CUDA_PATH_V12_9 / etc.)
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH") and v:
            m = re.search(r"(\d+)[._](\d+)", k) or re.search(r"(\d+)[._](\d+)", v)
            if m:
                parsed = (int(m.group(1)), int(m.group(2)))
                _detection_cache = (now, parsed)
                return parsed
    _detection_cache = (now, None)
    return None


def _map_cuda_to_channel(ver: tuple[int, int]) -> str:
    major, minor = ver
    if (major, minor) >= (13, 0):
        return "cu130"
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
        ver = detect_cuda_version()
        return _map_cuda_to_channel(ver) if ver else "cpu"
    # user-picked explicit channels (including rocm6.3 / cpu)
    return c if c in SUPPORTED_CHANNELS else "cpu"


def describe_choice(choice: str) -> tuple[str, str]:
    channel = normalize_choice(choice)
    selected = choice.lower().strip()
    if selected == "auto":
        ver = detect_cuda_version()
        if ver is None:
            return (channel, "Auto-detect found no CUDA runtime. Falling back to CPU.")
        return (
            channel,
            (
                f"Auto-detect found CUDA {ver[0]}.{ver[1]}; "
                f"selecting {channel.upper()} wheels."
            ),
        )
    if channel == "cpu":
        return (channel, "CPU-only wheels selected.")
    if channel.startswith("rocm"):
        return (channel, f"Manual selection: {channel.upper()} wheels.")
    return (channel, f"Manual selection: CUDA wheels for {channel.upper()}.")
