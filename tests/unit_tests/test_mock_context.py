import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4


def _load_mock_context_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "mock_context.py"
    module_name = f"addon_under_test.mock_context_{uuid4().hex}"

    addon_pkg = ModuleType("addon_under_test")
    addon_pkg.__path__ = [str(repo_root)]

    model_support = importlib.import_module("model_support")

    with patch.dict(
        sys.modules,
        {
            "addon_under_test": addon_pkg,
            "addon_under_test.model_support": model_support,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load mock_context module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_mock_scene_uses_sd15_defaults() -> None:
    mock_context = _load_mock_context_module()
    model_support = importlib.import_module("model_support")

    scene = mock_context.MockScene("sd15")
    defaults = model_support.get_default_model_paths("sd15")

    assert scene.sd_version == "sd15"
    assert scene.checkpoint_path == defaults["checkpoint_path"]
    assert scene.controlnet_union_path == defaults["controlnet_union_path"]
    assert scene.depth_controlnet_path == defaults["depth_controlnet_path"]
    assert scene.canny_controlnet_path == defaults["canny_controlnet_path"]
    assert scene.normal_controlnet_path == defaults["normal_controlnet_path"]


def test_mock_scene_uses_sdxl_defaults() -> None:
    mock_context = _load_mock_context_module()
    model_support = importlib.import_module("model_support")

    scene = mock_context.MockScene("sdxl")
    defaults = model_support.get_default_model_paths("sdxl")

    assert scene.sd_version == "sdxl"
    assert scene.checkpoint_path == defaults["checkpoint_path"]
    assert scene.controlnet_union_path == defaults["controlnet_union_path"]
    assert scene.depth_controlnet_path == defaults["depth_controlnet_path"]
    assert scene.canny_controlnet_path == defaults["canny_controlnet_path"]
    assert scene.normal_controlnet_path == defaults["normal_controlnet_path"]
