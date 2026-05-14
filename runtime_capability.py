from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import bpy

try:
    from .installer.cuda import normalize_choice
    from .installer.paths import deps_target_dir
except ImportError:
    from installer.cuda import normalize_choice
    from installer.paths import deps_target_dir

_PREFERRED_CYCLES_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")
_GPU_SCENE_DEVICE = "GPU"
_CPU_SCENE_DEVICE = "CPU"
_CYCLES_UI_GPU = "gpu"
_CYCLES_UI_CPU = "cpu"
_CYCLES_UI_INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class CyclesRenderSelection:
    """Resolved Cycles render-device selection for the current scene."""

    cycles_backend: str | None
    scene_render_device: str | None
    can_render: bool
    message: str


@dataclass(frozen=True, slots=True)
class CyclesUiCapability:
    """Non-mutating Cycles capability status for the preferences panel."""

    status: str
    cycles_backend: str | None
    scene_render_device: str | None
    message: str


@dataclass(frozen=True, slots=True)
class _DiffusionRuntimeProbe:
    device: str | None
    device_count: int | None
    primary_device_name: str | None
    torch_version: str | None
    torch_cuda_build: str | None


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    """Combined dependency and runtime capability for texture generation."""

    selected_torch_choice: str
    torch_install_channel: str
    active_deps_path: str | None
    torch_module_path: str | None
    torch_version: str | None
    torch_cuda_build: str | None
    diffusers_module_path: str | None
    diffusion_dependencies_importable: bool
    diffusion_environment_warning: str | None
    diffusion_device: str | None
    diffusion_device_count: int | None
    diffusion_primary_device_name: str | None
    cycles_ui_status: str
    cycles_backend: str | None
    scene_render_device: str | None
    can_generate: bool
    message: str


class _BlRnaOwner(Protocol):
    bl_rna: object


class _SceneRender(Protocol):
    engine: str


class _SceneCycles(Protocol):
    device: str
    bl_rna: object


class _SceneLike(Protocol):
    render: _SceneRender
    cycles: _SceneCycles


class _CyclesPreferences(Protocol):
    compute_device_type: str
    bl_rna: object
    devices: object


def _enum_identifiers(owner: _BlRnaOwner, property_name: str) -> set[str] | None:
    try:
        prop = owner.bl_rna.properties[property_name]
    except Exception:  # noqa: BLE001
        return None

    try:
        return {str(item.identifier) for item in prop.enum_items}
    except Exception:  # noqa: BLE001
        return None


def _set_render_engine(scene: _SceneLike, engine: str) -> bool:
    try:
        scene.render.engine = engine
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


def _set_scene_cycles_device(scene: _SceneLike, device: str) -> bool:
    available = _enum_identifiers(scene.cycles, "device")
    if available is not None and device not in available:
        return False

    try:
        scene.cycles.device = device
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


def _set_cycles_compute_device_type(
    preferences: _CyclesPreferences | None,
    backend: str,
) -> bool:
    if preferences is None:
        return False

    available = _enum_identifiers(preferences, "compute_device_type")
    if available is not None and backend not in available:
        return False

    try:
        preferences.compute_device_type = backend
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


def _refresh_cycles_devices(preferences: _CyclesPreferences) -> None:
    for method_name in ("refresh_devices", "get_devices"):
        method = getattr(preferences, method_name, None)
        if callable(method):
            with suppress(Exception):
                method()


def _flatten_cycles_devices(devices: object) -> list[object]:
    if devices is None:
        return []

    if isinstance(devices, dict):
        return list(devices.values())

    if isinstance(devices, Iterable):
        flattened = list(devices)
    else:
        return []

    result: list[object] = []
    for item in flattened:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten_cycles_devices(item))
            continue
        result.append(item)
    return result


def _backend_has_enabled_device(
    preferences: _CyclesPreferences | None,
    backend: str,
) -> bool:
    if preferences is None:
        return False

    _refresh_cycles_devices(preferences)
    devices = _flatten_cycles_devices(getattr(preferences, "devices", None))
    if not devices:
        return False

    matching = [
        device
        for device in devices
        if str(getattr(device, "type", "")).upper() == backend
    ]
    if not matching:
        return False

    return any(bool(getattr(device, "use", True)) for device in matching)


def _restore_cycles_selection(
    scene: _SceneLike,
    preferences: _CyclesPreferences | None,
    *,
    original_engine: str | None,
    original_scene_device: str | None,
    original_compute_device: str | None,
) -> None:
    if original_engine is not None:
        _set_render_engine(scene, original_engine)
    if original_scene_device is not None:
        _set_scene_cycles_device(scene, original_scene_device)
    if preferences is not None and original_compute_device is not None:
        with suppress(Exception):
            preferences.compute_device_type = original_compute_device


def _get_cycles_preferences() -> _CyclesPreferences | None:
    try:
        return bpy.context.preferences.addons["cycles"].preferences
    except Exception:  # noqa: BLE001
        return None


def _get_selected_torch_choice(explicit_choice: str | None = None) -> str:
    if explicit_choice is not None:
        return str(explicit_choice)

    try:
        return str(bpy.context.preferences.addons[__package__].preferences.cuda_variant)
    except Exception:  # noqa: BLE001
        return "AUTO"


def resolve_torch_install_channel(choice: str | None = None) -> str:
    return normalize_choice(_get_selected_torch_choice(choice))


def _module_file_path(module: object | None) -> str | None:
    if module is None:
        return None
    value = getattr(module, "__file__", None)
    if not value:
        return None
    return str(value)


def _path_key(path_value: str) -> str:
    try:
        return os.path.normcase(str(Path(path_value).resolve()))
    except Exception:  # noqa: BLE001
        return os.path.normcase(os.path.normpath(path_value))


def _module_from_active_env(
    module_path: str | None,
    active_deps_path: str | None,
) -> bool | None:
    if module_path is None or active_deps_path is None:
        return None
    module_key = _path_key(module_path)
    active_key = _path_key(active_deps_path)
    return module_key.startswith(active_key)


def _probe_diffusion_runtime(torch_module: object | None) -> _DiffusionRuntimeProbe:
    if torch_module is None:
        return _DiffusionRuntimeProbe(
            device=None,
            device_count=None,
            primary_device_name=None,
            torch_version=None,
            torch_cuda_build=None,
        )

    torch_version = str(getattr(torch_module, "__version__", None) or "")
    torch_version_value = torch_version or None
    torch_cuda_build = getattr(getattr(torch_module, "version", None), "cuda", None)
    torch_cuda_build_value = (
        str(torch_cuda_build)
        if torch_cuda_build is not None
        else None
    )

    try:
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            device_count = int(torch_module.cuda.device_count())
            device_name = None
            if device_count > 0:
                with suppress(Exception):
                    device_name = str(torch_module.cuda.get_device_name(0))
            return _DiffusionRuntimeProbe(
                device="cuda",
                device_count=device_count,
                primary_device_name=device_name,
                torch_version=torch_version_value,
                torch_cuda_build=torch_cuda_build_value,
            )
        if (
            getattr(torch_module.backends, "mps", None)
            and torch_module.backends.mps.is_available()
        ):
            return _DiffusionRuntimeProbe(
                device="mps",
                device_count=None,
                primary_device_name=None,
                torch_version=torch_version_value,
                torch_cuda_build=torch_cuda_build_value,
            )
    except Exception:  # noqa: BLE001
        return _DiffusionRuntimeProbe(
            device="cpu",
            device_count=None,
            primary_device_name=None,
            torch_version=torch_version_value,
            torch_cuda_build=torch_cuda_build_value,
        )

    return _DiffusionRuntimeProbe(
        device="cpu",
        device_count=None,
        primary_device_name=None,
        torch_version=torch_version_value,
        torch_cuda_build=torch_cuda_build_value,
    )


def resolve_diffusion_device(torch_module: object | None = None) -> str | None:
    torch = torch_module
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except Exception:  # noqa: BLE001
            return None
    return _probe_diffusion_runtime(torch).device


def _import_optional_module(module_name: str) -> object | None:
    try:
        return importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return None


def _cycles_preferences_summary(preferences: _CyclesPreferences | None) -> str:
    if preferences is None:
        return "no Cycles preferences available"

    available = _enum_identifiers(preferences, "compute_device_type")
    available_text = ", ".join(sorted(available)) if available else "<unknown>"

    _refresh_cycles_devices(preferences)
    devices = _flatten_cycles_devices(getattr(preferences, "devices", None))
    if not devices:
        devices_text = "<none>"
    else:
        device_parts = []
        for device in devices:
            dtype = str(getattr(device, "type", "")).upper() or "UNKNOWN"
            enabled = bool(getattr(device, "use", True))
            marker = "on" if enabled else "off"
            device_parts.append(f"{dtype}:{marker}")
        devices_text = ", ".join(device_parts)

    return f"available backends [{available_text}], devices [{devices_text}]"


def _select_cycles_gpu_backend(
    scene: _SceneLike,
    preferences: _CyclesPreferences | None,
) -> CyclesRenderSelection | None:
    for backend in _PREFERRED_CYCLES_BACKENDS:
        if not _set_cycles_compute_device_type(preferences, backend):
            continue
        if not _backend_has_enabled_device(preferences, backend):
            continue
        if _set_scene_cycles_device(scene, _GPU_SCENE_DEVICE):
            return CyclesRenderSelection(
                cycles_backend=backend,
                scene_render_device=_GPU_SCENE_DEVICE,
                can_render=True,
                message=f"Cycles render backend: {backend} (GPU).",
            )
    return None


def probe_cycles_ui_capability(
    context: bpy.types.Context,
) -> CyclesUiCapability:
    scene = getattr(context, "scene", None)
    preferences = _get_cycles_preferences()
    if preferences is None:
        return CyclesUiCapability(
            status=_CYCLES_UI_INCONCLUSIVE,
            cycles_backend=None,
            scene_render_device=None,
            message=(
                "Cycles capability is inconclusive in preferences; "
                "render-time setup will choose a device."
            ),
        )

    _refresh_cycles_devices(preferences)
    devices = _flatten_cycles_devices(getattr(preferences, "devices", None))
    if not devices:
        return CyclesUiCapability(
            status=_CYCLES_UI_INCONCLUSIVE,
            cycles_backend=None,
            scene_render_device=None,
            message=(
                "Cycles capability is inconclusive in preferences; "
                "device list is unavailable."
            ),
        )

    for backend in _PREFERRED_CYCLES_BACKENDS:
        matching = [
            device
            for device in devices
            if str(getattr(device, "type", "")).upper() == backend
        ]
        if matching and any(bool(getattr(device, "use", True)) for device in matching):
            return CyclesUiCapability(
                status=_CYCLES_UI_GPU,
                cycles_backend=backend,
                scene_render_device=_GPU_SCENE_DEVICE,
                message=f"Cycles UI capability: {backend} (GPU).",
            )

    has_gpu_devices = any(
        str(getattr(device, "type", "")).upper() in _PREFERRED_CYCLES_BACKENDS
        for device in devices
    )
    if has_gpu_devices:
        return CyclesUiCapability(
            status=_CYCLES_UI_CPU,
            cycles_backend=None,
            scene_render_device=_CPU_SCENE_DEVICE,
            message="Cycles UI capability: CPU (GPU devices are disabled).",
        )

    scene_device = str(getattr(getattr(scene, "cycles", None), "device", "")).upper()
    if scene_device == _CPU_SCENE_DEVICE:
        return CyclesUiCapability(
            status=_CYCLES_UI_CPU,
            cycles_backend=None,
            scene_render_device=_CPU_SCENE_DEVICE,
            message="Cycles UI capability: CPU.",
        )

    return CyclesUiCapability(
        status=_CYCLES_UI_INCONCLUSIVE,
        cycles_backend=None,
        scene_render_device=None,
        message=(
            "Cycles capability is inconclusive in preferences; "
            "render-time setup will choose a device."
        ),
    )


def resolve_cycles_render_selection(
    context: bpy.types.Context,
    *,
    apply: bool = False,
) -> CyclesRenderSelection:
    scene = context.scene
    preferences = _get_cycles_preferences()

    original_engine = getattr(getattr(scene, "render", None), "engine", None)
    original_scene_device = getattr(getattr(scene, "cycles", None), "device", None)
    original_compute_device = getattr(preferences, "compute_device_type", None)
    summary = _cycles_preferences_summary(preferences)

    try:
        if not _set_render_engine(scene, "CYCLES"):
            return CyclesRenderSelection(
                cycles_backend=None,
                scene_render_device=None,
                can_render=False,
                message=(
                    "Cycles rendering is unavailable in this Blender session. "
                    f"Probe: {summary}."
                ),
            )

        gpu_selection = _select_cycles_gpu_backend(scene, preferences)
        if gpu_selection is not None:
            return gpu_selection

        if _set_scene_cycles_device(scene, _CPU_SCENE_DEVICE):
            return CyclesRenderSelection(
                cycles_backend=None,
                scene_render_device=_CPU_SCENE_DEVICE,
                can_render=True,
                message="Cycles render backend: CPU.",
            )

        return CyclesRenderSelection(
            cycles_backend=None,
            scene_render_device=None,
            can_render=False,
            message=(
                "Cycles could not configure a usable render device. "
                f"Probe: {summary}."
            ),
        )
    finally:
        if not apply:
            _restore_cycles_selection(
                scene,
                preferences,
                original_engine=original_engine,
                original_scene_device=original_scene_device,
                original_compute_device=original_compute_device,
            )


def configure_cycles_render_device(
    context: bpy.types.Context,
) -> CyclesRenderSelection:
    selection = resolve_cycles_render_selection(context, apply=True)
    if not selection.can_render:
        raise SystemError(selection.message)
    return selection


def _active_deps_path() -> str | None:
    try:
        return str(Path(deps_target_dir()).resolve())
    except Exception:  # noqa: BLE001
        return None


def _environment_mismatch_warning(
    *,
    active_deps_path: str | None,
    torch_module_path: str | None,
    diffusers_module_path: str | None,
) -> str | None:
    torch_from_active = _module_from_active_env(torch_module_path, active_deps_path)
    diffusers_from_active = _module_from_active_env(
        diffusers_module_path,
        active_deps_path,
    )
    if torch_from_active is True and diffusers_from_active is True:
        return None

    if (
        torch_from_active is None
        and diffusers_from_active is None
    ):
        return None

    if active_deps_path is None:
        return None

    return (
        "Imported diffusion modules do not match the active addon environment. "
        f"Active env: {active_deps_path}. "
        f"torch: {torch_module_path or '<unavailable>'}. "
        f"diffusers: {diffusers_module_path or '<unavailable>'}."
    )


def get_runtime_capability(
    context: bpy.types.Context,
    *,
    torch_choice: str | None = None,
) -> RuntimeCapability:
    selected_torch_choice = _get_selected_torch_choice(torch_choice)
    torch_install_channel = resolve_torch_install_channel(selected_torch_choice)
    active_deps_path = _active_deps_path()

    torch_module = _import_optional_module("torch")
    diffusers_module = _import_optional_module("diffusers")
    diffusion_dependencies_importable = (
        torch_module is not None
        and diffusers_module is not None
    )

    torch_module_path = _module_file_path(torch_module)
    diffusers_module_path = _module_file_path(diffusers_module)
    diffusion_probe = _probe_diffusion_runtime(torch_module)
    cycles_ui = probe_cycles_ui_capability(context)
    mismatch_warning = _environment_mismatch_warning(
        active_deps_path=active_deps_path,
        torch_module_path=torch_module_path,
        diffusers_module_path=diffusers_module_path,
    )

    can_generate = (
        diffusion_dependencies_importable
        and diffusion_probe.device is not None
    )

    parts = [
        (
            "Selected dependency backend: "
            f"{selected_torch_choice} -> {torch_install_channel}."
        ),
        cycles_ui.message,
    ]
    if not diffusion_dependencies_importable:
        parts.append(
            "Diffusion dependencies are not importable. "
            "Install Python Dependencies and restart Blender.",
        )
    elif diffusion_probe.device is None:
        parts.append("Diffusion device: unavailable.")
    else:
        diffusion_detail = f"Diffusion device: {diffusion_probe.device}."
        if (
            diffusion_probe.device == "cuda"
            and diffusion_probe.device_count is not None
        ):
            diffusion_detail = (
                f"{diffusion_detail} CUDA devices: {diffusion_probe.device_count}."
            )
        if diffusion_probe.primary_device_name:
            diffusion_detail = (
                f"{diffusion_detail} Primary GPU: "
                f"{diffusion_probe.primary_device_name}."
            )
        parts.append(diffusion_detail)

    if mismatch_warning:
        parts.append(mismatch_warning)

    return RuntimeCapability(
        selected_torch_choice=selected_torch_choice,
        torch_install_channel=torch_install_channel,
        active_deps_path=active_deps_path,
        torch_module_path=torch_module_path,
        torch_version=diffusion_probe.torch_version,
        torch_cuda_build=diffusion_probe.torch_cuda_build,
        diffusers_module_path=diffusers_module_path,
        diffusion_dependencies_importable=diffusion_dependencies_importable,
        diffusion_environment_warning=mismatch_warning,
        diffusion_device=diffusion_probe.device,
        diffusion_device_count=diffusion_probe.device_count,
        diffusion_primary_device_name=diffusion_probe.primary_device_name,
        cycles_ui_status=cycles_ui.status,
        cycles_backend=cycles_ui.cycles_backend,
        scene_render_device=cycles_ui.scene_render_device,
        can_generate=can_generate,
        message=" ".join(parts),
    )
