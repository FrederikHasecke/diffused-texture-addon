from dataclasses import dataclass
from pathlib import Path

SUPPORTED_PYTHON_SERIES = {
    (3, 11): "py311",
    (3, 13): "py313",
}
_CONSTRAINTS_DIR = Path(__file__).with_name("constraints")


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Resolved runtime dependency selection for the current Blender/Python pair."""

    blender_version: tuple[int, int, int]
    python_version: tuple[int, int]
    python_tag: str
    constraints_path: Path
    runtime_requirements: tuple[str, ...]


def expected_python_version(
    blender_version: tuple[int, int, int],
) -> tuple[int, int]:
    if blender_version < (5, 1, 0):
        return (3, 11)
    return (3, 13)


def normalize_python_version(python_version: tuple[int, int]) -> tuple[int, int]:
    version = tuple(python_version[:2])
    if version not in SUPPORTED_PYTHON_SERIES:
        supported = ", ".join(
            f"{major}.{minor}" for major, minor in SUPPORTED_PYTHON_SERIES
        )
        msg = (
            f"Unsupported Python version {version[0]}.{version[1]}. "
            f"Supported versions: {supported}."
        )
        raise ValueError(msg)
    return version


def constraints_filename(python_version: tuple[int, int]) -> str:
    normalized = normalize_python_version(python_version)
    return f"runtime-{SUPPORTED_PYTHON_SERIES[normalized]}.txt"


def read_requirements_file(requirements_path: Path) -> list[str]:
    if not requirements_path.exists():
        msg = f"Requirements file not found: {requirements_path}"
        raise FileNotFoundError(msg)

    requirements: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)

    if not requirements:
        msg = f"Requirements file is empty: {requirements_path}"
        raise ValueError(msg)
    return requirements


def resolve_runtime_spec(
    *,
    blender_version: tuple[int, int, int],
    python_version: tuple[int, int],
    constraints_dir: Path | None = None,
) -> RuntimeSpec:
    normalized_python = normalize_python_version(python_version)
    python_tag = SUPPORTED_PYTHON_SERIES[normalized_python]
    root = constraints_dir if constraints_dir is not None else _CONSTRAINTS_DIR
    constraints_path = root / constraints_filename(normalized_python)
    requirements = tuple(read_requirements_file(constraints_path))
    return RuntimeSpec(
        blender_version=blender_version,
        python_version=normalized_python,
        python_tag=python_tag,
        constraints_path=constraints_path,
        runtime_requirements=requirements,
    )


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
                "ROCm wheels were requested on Windows and may not be available.",
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
    return (f"https://download.pytorch.org/whl/{channel}", f"PyTorch {channel.upper()}")
