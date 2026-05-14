import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest


def _enum_property(*identifiers: str) -> SimpleNamespace:
    return SimpleNamespace(
        enum_items=[SimpleNamespace(identifier=identifier) for identifier in identifiers]
    )


def _make_context() -> SimpleNamespace:
    scene = SimpleNamespace(
        render=SimpleNamespace(engine="BLENDER_EEVEE"),
        cycles=SimpleNamespace(
            device="CPU",
            bl_rna=SimpleNamespace(
                properties={"device": _enum_property("CPU", "GPU")},
            ),
        ),
    )
    return SimpleNamespace(scene=scene)


def _load_runtime_capability(
    *,
    available_backends: tuple[str, ...],
    device_uses: dict[str, bool] | None,
    active_deps_path: str = "C:/deps/env_active",
) -> tuple[ModuleType, ModuleType, SimpleNamespace, SimpleNamespace]:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "runtime_capability.py"
    package_name = f"addon_under_test_{uuid4().hex}"
    module_name = f"{package_name}.runtime_capability"

    addon_pkg = ModuleType(package_name)
    addon_pkg.__path__ = [str(repo_root)]

    installer_pkg = ModuleType(f"{package_name}.installer")
    installer_pkg.__path__ = [str(repo_root / "installer")]

    cuda = ModuleType(f"{package_name}.installer.cuda")
    cuda.normalize_choice = lambda choice: "cu130" if str(choice).upper() == "AUTO" else str(choice).lower()

    paths = ModuleType(f"{package_name}.installer.paths")
    paths.deps_target_dir = lambda: Path(active_deps_path)

    devices = []
    if device_uses is not None:
        devices = [
            SimpleNamespace(type=backend, use=enabled)
            for backend, enabled in device_uses.items()
        ]

    cycles_preferences = SimpleNamespace(
        compute_device_type="CUDA",
        devices=devices,
        bl_rna=SimpleNamespace(
            properties={
                "compute_device_type": _enum_property(*available_backends),
            },
        ),
    )
    cycles_preferences.get_devices = lambda: None

    addon_preferences = SimpleNamespace(cuda_variant="AUTO")

    bpy_module = ModuleType("bpy")
    bpy_module.types = SimpleNamespace(Context=object)
    bpy_module.context = SimpleNamespace(
        preferences=SimpleNamespace(
            addons={
                "cycles": SimpleNamespace(preferences=cycles_preferences),
                package_name: SimpleNamespace(preferences=addon_preferences),
            },
        ),
    )

    with patch.dict(
        sys.modules,
        {
            package_name: addon_pkg,
            f"{package_name}.installer": installer_pkg,
            f"{package_name}.installer.cuda": cuda,
            f"{package_name}.installer.paths": paths,
            "bpy": bpy_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load runtime_capability module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        context = _make_context()
        return module, bpy_module, context, cycles_preferences


def _torch_module(
    *,
    module_path: str,
    cuda_available: bool = True,
    device_name: str = "NVIDIA GeForce RTX 3080 Ti",
    version: str = "2.12.0+cu130",
) -> object:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def device_count() -> int:
            return 1 if cuda_available else 0

        @staticmethod
        def get_device_name(index: int) -> str:  # noqa: ARG004
            return device_name

    return SimpleNamespace(
        __file__=module_path,
        __version__=version,
        cuda=_Cuda(),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        version=SimpleNamespace(cuda="13.0"),
    )


def test_resolve_cycles_render_selection_prefers_optix_over_cuda() -> None:
    runtime_capability, _, context, cycles_preferences = _load_runtime_capability(
        available_backends=("OPTIX", "CUDA"),
        device_uses={"OPTIX": True, "CUDA": True},
    )

    selection = runtime_capability.resolve_cycles_render_selection(context, apply=False)

    assert selection.cycles_backend == "OPTIX"
    assert selection.scene_render_device == "GPU"
    assert context.scene.render.engine == "BLENDER_EEVEE"
    assert context.scene.cycles.device == "CPU"
    assert cycles_preferences.compute_device_type == "CUDA"


def test_probe_cycles_ui_capability_reports_inconclusive_when_devices_missing() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses=None,
    )

    capability = runtime_capability.probe_cycles_ui_capability(context)

    assert capability.status == "inconclusive"
    assert "inconclusive" in capability.message.lower()


def test_resolve_cycles_render_selection_falls_back_to_cpu_when_devices_missing() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses=None,
    )

    selection = runtime_capability.resolve_cycles_render_selection(context, apply=False)

    assert selection.cycles_backend is None
    assert selection.scene_render_device == "CPU"
    assert selection.can_render
    assert "CPU" in selection.message


def test_resolve_cycles_render_selection_falls_back_to_cpu_when_backend_missing() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses={"CPU": True},
    )

    selection = runtime_capability.resolve_cycles_render_selection(context, apply=False)

    assert selection.cycles_backend is None
    assert selection.scene_render_device == "CPU"
    assert selection.can_render
    assert "CPU" in selection.message


def test_resolve_cycles_render_selection_failure_includes_probe_details() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses={"CUDA": True},
    )

    class _RenderBlocked:
        def __init__(self) -> None:
            self._engine = "BLENDER_EEVEE"

        @property
        def engine(self) -> str:
            return self._engine

        @engine.setter
        def engine(self, value: str) -> None:  # noqa: ARG002
            msg = "blocked"
            raise RuntimeError(msg)

    context.scene.render = _RenderBlocked()
    selection = runtime_capability.resolve_cycles_render_selection(context, apply=False)

    assert not selection.can_render
    assert "Probe:" in selection.message


def test_get_runtime_capability_reports_cuda_runtime_from_active_env() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses={"CUDA": True},
    )
    torch_module = _torch_module(
        module_path="C:/deps/env_active/torch/__init__.py",
    )
    diffusers_module = SimpleNamespace(__file__="C:/deps/env_active/diffusers/__init__.py")

    with patch.object(
        runtime_capability.importlib,
        "import_module",
        side_effect=lambda name: (
            torch_module if name == "torch" else diffusers_module
        ),
    ):
        capability = runtime_capability.get_runtime_capability(
            context,
            torch_choice="AUTO",
        )

    assert capability.selected_torch_choice == "AUTO"
    assert capability.torch_install_channel == "cu130"
    assert capability.diffusion_dependencies_importable
    assert capability.diffusion_device == "cuda"
    assert capability.diffusion_device_count == 1
    assert capability.diffusion_primary_device_name == "NVIDIA GeForce RTX 3080 Ti"
    assert capability.diffusion_environment_warning is None


def test_get_runtime_capability_reports_environment_mismatch_warning() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses={"CUDA": True},
        active_deps_path="C:/deps/env_active",
    )
    torch_module = _torch_module(
        module_path="C:/external_env/site-packages/torch/__init__.py",
    )
    diffusers_module = SimpleNamespace(
        __file__="C:/external_env/site-packages/diffusers/__init__.py"
    )

    with patch.object(
        runtime_capability.importlib,
        "import_module",
        side_effect=lambda name: (
            torch_module if name == "torch" else diffusers_module
        ),
    ):
        capability = runtime_capability.get_runtime_capability(
            context,
            torch_choice="cu130",
        )

    assert capability.torch_install_channel == "cu130"
    assert capability.diffusion_environment_warning is not None
    assert "Active env:" in capability.diffusion_environment_warning
    assert "env_active" in capability.diffusion_environment_warning
    assert "external_env" in capability.diffusion_environment_warning


def test_get_runtime_capability_reports_missing_dependencies() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses={"CUDA": False},
    )

    def _import_module(name: str):  # noqa: ANN001
        if name == "torch":
            return _torch_module(
                module_path="C:/deps/env_active/torch/__init__.py",
                cuda_available=False,
                version="2.12.0+cpu",
            )
        msg = f"No module named {name}"
        raise ModuleNotFoundError(msg)

    with patch.object(runtime_capability.importlib, "import_module", side_effect=_import_module):
        capability = runtime_capability.get_runtime_capability(
            context,
            torch_choice="rocm6.3",
        )

    assert capability.torch_install_channel == "rocm6.3"
    assert not capability.diffusion_dependencies_importable
    assert capability.diffusion_device == "cpu"
    assert not capability.can_generate
    assert "Install Python Dependencies and restart Blender." in capability.message
