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
    device_uses: dict[str, bool],
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
    cuda.normalize_choice = lambda choice: choice.lower()

    cycles_preferences = SimpleNamespace(
        compute_device_type="CUDA",
        devices=[
            SimpleNamespace(type=backend, use=enabled)
            for backend, enabled in device_uses.items()
        ],
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


@pytest.mark.parametrize("backend", ["HIP", "ONEAPI", "METAL"])
def test_resolve_cycles_render_selection_accepts_other_gpu_backends(
    backend: str,
) -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=(backend,),
        device_uses={backend: True},
    )

    selection = runtime_capability.resolve_cycles_render_selection(context, apply=False)

    assert selection.cycles_backend == backend
    assert selection.scene_render_device == "GPU"
    assert selection.can_render


def test_resolve_cycles_render_selection_falls_back_to_cpu() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA", "HIP"),
        device_uses={"CUDA": False, "HIP": False},
    )

    selection = runtime_capability.resolve_cycles_render_selection(context, apply=False)

    assert selection.cycles_backend is None
    assert selection.scene_render_device == "CPU"
    assert selection.can_render
    assert selection.message == "Cycles render backend: CPU."


def test_get_runtime_capability_reports_install_target_and_missing_dependencies() -> None:
    runtime_capability, _, context, _ = _load_runtime_capability(
        available_backends=("CUDA",),
        device_uses={"CUDA": False},
    )

    def _import_module(name: str):  # noqa: ANN001
        if name == "torch":
            return object()
        msg = f"No module named {name}"
        raise ModuleNotFoundError(msg)

    with patch.object(runtime_capability.importlib, "import_module", side_effect=_import_module):
        capability = runtime_capability.get_runtime_capability(
            context,
            torch_choice="rocm6.3",
        )

    assert capability.torch_install_channel == "rocm6.3"
    assert capability.cycles_backend is None
    assert capability.scene_render_device == "CPU"
    assert capability.diffusion_device is None
    assert not capability.can_generate
    assert "Dependency backend: rocm6.3." in capability.message
    assert "Cycles render: CPU." in capability.message
    assert "Install Python Dependencies and restart Blender." in capability.message
