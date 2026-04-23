from __future__ import annotations

import importlib
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import bpy

try:
    from .installer.cuda import normalize_choice
except ImportError:
    from installer.cuda import normalize_choice

_PREFERRED_CYCLES_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")
_GPU_SCENE_DEVICE = "GPU"
_CPU_SCENE_DEVICE = "CPU"


@dataclass(frozen=True, slots=True)
class CyclesRenderSelection:
    """Resolved Cycles render-device selection for the current scene."""

    cycles_backend: str | None
    scene_render_device: str | None
    can_render: bool
    message: str


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    """Combined dependency and runtime capability for texture generation."""

    torch_install_channel: str
    cycles_backend: str | None
    scene_render_device: str | None
    diffusion_device: str | None
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
        return True

    matching = [
        device
        for device in devices
        if str(getattr(device, "type", "")).upper() == backend
    ]
    if not matching:
        return True

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
        return explicit_choice

    try:
        return str(bpy.context.preferences.addons[__package__].preferences.cuda_variant)
    except Exception:  # noqa: BLE001
        return "AUTO"


def resolve_torch_install_channel(choice: str | None = None) -> str:
    return normalize_choice(_get_selected_torch_choice(choice))


def resolve_diffusion_device(torch_module: object | None = None) -> str | None:
    torch = torch_module
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except Exception:  # noqa: BLE001
            return None

    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cpu"


def _diffusion_dependencies_ready() -> bool:
    for module_name in ("torch", "diffusers"):
        try:
            importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            return False
    return True


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

    try:
        if not _set_render_engine(scene, "CYCLES"):
            return CyclesRenderSelection(
                cycles_backend=None,
                scene_render_device=None,
                can_render=False,
                message="Cycles rendering is unavailable in this Blender session.",
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
            message="Cycles could not configure a usable render device.",
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


def get_runtime_capability(
    context: bpy.types.Context,
    *,
    torch_choice: str | None = None,
) -> RuntimeCapability:
    torch_install_channel = resolve_torch_install_channel(torch_choice)
    cycles = resolve_cycles_render_selection(context, apply=False)
    dependencies_ready = _diffusion_dependencies_ready()
    diffusion_device = resolve_diffusion_device() if dependencies_ready else None

    can_generate = cycles.can_render and diffusion_device is not None

    parts = [f"Dependency backend: {torch_install_channel}."]
    if cycles.scene_render_device == _CPU_SCENE_DEVICE:
        parts.append("Cycles render: CPU.")
    elif cycles.cycles_backend and cycles.scene_render_device:
        parts.append(
            f"Cycles render: {cycles.cycles_backend} ({cycles.scene_render_device}).",
        )
    else:
        parts.append(cycles.message)

    if diffusion_device is None:
        parts.append(
            "Diffusion dependencies are not importable. "
            "Install Python Dependencies and restart Blender.",
        )
    else:
        parts.append(f"Diffusion device: {diffusion_device}.")

    return RuntimeCapability(
        torch_install_channel=torch_install_channel,
        cycles_backend=cycles.cycles_backend,
        scene_render_device=cycles.scene_render_device,
        diffusion_device=diffusion_device,
        can_generate=can_generate,
        message=" ".join(parts),
    )
