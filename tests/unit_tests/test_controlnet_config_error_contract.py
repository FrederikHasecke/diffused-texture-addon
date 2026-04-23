import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest


def _load_controlnet_config() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "pipeline" / "controlnet_config.py"
    module_name = (
        "diffused_texture_addon.diffusedtexture.pipeline.controlnet_config_"
        f"{uuid4().hex}"
    )

    addon_pkg = ModuleType("diffused_texture_addon")
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType("diffused_texture_addon.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    pipeline_pkg = ModuleType("diffused_texture_addon.diffusedtexture.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "diffusedtexture" / "pipeline")]

    blender_ops = ModuleType("diffused_texture_addon.blender_operations")
    blender_ops.ProcessParameter = object

    model_support = ModuleType("diffused_texture_addon.model_support")
    model_support.require_supported_sd_version = lambda sd_version: sd_version

    with patch.dict(
        sys.modules,
        {
            "diffused_texture_addon": addon_pkg,
            "diffused_texture_addon.diffusedtexture": diffusedtexture_pkg,
            "diffused_texture_addon.diffusedtexture.pipeline": pipeline_pkg,
            "diffused_texture_addon.blender_operations": blender_ops,
            "diffused_texture_addon.model_support": model_support,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load controlnet_config module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _dummy_process_parameter() -> SimpleNamespace:
    return SimpleNamespace(
        operation_mode="PARALLEL_IMG",
        sd_version="sd15",
        mesh_complexity="HIGH",
        dtype="float16",
        controlnet_union_path="union",
        depth_controlnet_path="depth",
        canny_controlnet_path="canny",
        normal_controlnet_path="normal",
    )


def test_load_controlnet_models_uses_runtime_imports_after_module_import() -> None:
    controlnet_config = _load_controlnet_config()

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"

    class _FakeControlNetModel:
        calls: list[tuple[str, str]] = []

        @classmethod
        def from_pretrained(cls, path: str, torch_dtype: str) -> dict[str, str]:
            cls.calls.append((path, torch_dtype))
            return {"path": path, "torch_dtype": torch_dtype}

    class _FakeControlNetUnionModel:
        @classmethod
        def from_pretrained(cls, path: str, torch_dtype: str) -> dict[str, str]:
            return {"path": path, "torch_dtype": torch_dtype}

    diffusers = ModuleType("diffusers")
    diffusers.ControlNetModel = _FakeControlNetModel
    diffusers.ControlNetUnionModel = _FakeControlNetUnionModel

    with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": diffusers}):
        models = controlnet_config.load_controlnet_models(_dummy_process_parameter())

    assert models == [  # noqa: S101
        {"path": "depth", "torch_dtype": "float16"},
        {"path": "canny", "torch_dtype": "float16"},
        {"path": "normal", "torch_dtype": "float16"},
    ]
    assert _FakeControlNetModel.calls == [  # noqa: S101
        ("depth", "float16"),
        ("canny", "float16"),
        ("normal", "float16"),
    ]


def test_load_controlnet_models_raises_restart_guidance_when_runtime_missing() -> None:
    controlnet_config = _load_controlnet_config()

    real_import = __import__

    def _import_without_runtime(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name in {"torch", "diffusers"}:
            msg = f"No module named '{name}'"
            raise ModuleNotFoundError(msg)
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch(
            "builtins.__import__",
            side_effect=_import_without_runtime,
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        controlnet_config.load_controlnet_models(_dummy_process_parameter())

    assert "restart Blender" in str(exc_info.value)  # noqa: S101
    assert "Install Python Dependencies" in str(exc_info.value)  # noqa: S101
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)  # noqa: S101
