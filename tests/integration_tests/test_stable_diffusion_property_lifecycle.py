import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

_SD_PROP_NAMES = (
    "sd_version",
    "checkpoint_path",
    "dtype",
    "custom_sd_resolution",
)


def _clear_sd_props(scene_type: type) -> None:
    for prop_name in _SD_PROP_NAMES:
        if hasattr(scene_type, prop_name):
            delattr(scene_type, prop_name)


def _load_stable_diffusion_module() -> tuple[ModuleType, type]:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "properties" / "stable_diffusion.py"
    module_name = f"addon_under_test.properties.stable_diffusion_{uuid4().hex}"

    addon_pkg = ModuleType("addon_under_test")
    addon_pkg.__path__ = [str(repo_root)]

    properties_pkg = ModuleType("addon_under_test.properties")
    properties_pkg.__path__ = [str(repo_root / "properties")]

    scene_type = type("Scene", (), {})

    bpy_module = ModuleType("bpy")
    bpy_module.types = SimpleNamespace(Scene=scene_type, Context=object)

    props_module = ModuleType("bpy.props")
    props_module.EnumProperty = lambda **kwargs: kwargs
    props_module.IntProperty = lambda **kwargs: kwargs
    props_module.StringProperty = lambda **kwargs: kwargs

    model_support = importlib.import_module("model_support")

    with patch.dict(
        sys.modules,
        {
            "addon_under_test": addon_pkg,
            "addon_under_test.properties": properties_pkg,
            "addon_under_test.model_support": model_support,
            "bpy": bpy_module,
            "bpy.props": props_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load stable_diffusion module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, scene_type


def test_stable_diffusion_register_unregister_cleans_scene_props() -> None:
    stable_diffusion, scene_type = _load_stable_diffusion_module()

    _clear_sd_props(scene_type)
    try:
        stable_diffusion.register_stable_diffusion_properties()
        for prop_name in _SD_PROP_NAMES:
            assert hasattr(scene_type, prop_name)

        stable_diffusion.unregister_stable_diffusion_properties()
        for prop_name in _SD_PROP_NAMES:
            assert not hasattr(scene_type, prop_name)
    finally:
        _clear_sd_props(scene_type)


def test_stable_diffusion_unregister_handles_partial_registration() -> None:
    stable_diffusion, scene_type = _load_stable_diffusion_module()

    _clear_sd_props(scene_type)
    try:
        stable_diffusion.register_stable_diffusion_properties()

        # Simulate partial registration cleanup path where one property is missing.
        if hasattr(scene_type, "dtype"):
            delattr(scene_type, "dtype")

        stable_diffusion.unregister_stable_diffusion_properties()
        for prop_name in _SD_PROP_NAMES:
            assert not hasattr(scene_type, prop_name)
    finally:
        _clear_sd_props(scene_type)
