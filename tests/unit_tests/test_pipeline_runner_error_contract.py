import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image


def _load_pipeline_runner() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "pipeline" / "pipeline_runner.py"
    module_name = (
        f"diffused_texture_addon.diffusedtexture.pipeline.pipeline_runner_{uuid4().hex}"
    )

    addon_pkg = ModuleType("diffused_texture_addon")
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType("diffused_texture_addon.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    pipeline_pkg = ModuleType("diffused_texture_addon.diffusedtexture.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "diffusedtexture" / "pipeline")]

    blender_ops = ModuleType("diffused_texture_addon.blender_operations")
    blender_ops.ProcessParameter = object

    controlnet_config = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.controlnet_config",
    )
    controlnet_config.get_controlnet_static_config = lambda process_parameter: {
        "inputs": ["depth"],
        "conditioning_scale": [1.0],
    }

    diffusers = ModuleType("diffusers")
    diffusers.StableDiffusionControlNetInpaintPipeline = object
    diffusers.StableDiffusionXLControlNetUnionInpaintPipeline = object

    with patch.dict(
        sys.modules,
        {
            "diffused_texture_addon": addon_pkg,
            "diffused_texture_addon.diffusedtexture": diffusedtexture_pkg,
            "diffused_texture_addon.diffusedtexture.pipeline": pipeline_pkg,
            "diffused_texture_addon.blender_operations": blender_ops,
            "diffused_texture_addon.diffusedtexture.pipeline.controlnet_config": controlnet_config,
            "diffusers": diffusers,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load pipeline_runner module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _dummy_process_parameter() -> SimpleNamespace:
    return SimpleNamespace(
        mesh_complexity="LOW",
        sd_version="sd15",
        my_prompt="prompt",
        my_negative_prompt="",
        use_ipadapter=False,
        ipadapter_image=None,
        texture_seed=0,
    )


def _dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))


def test_run_pipeline_raises_clear_error_when_torch_missing() -> None:
    pipeline_runner = _load_pipeline_runner()

    class _NoopPipe:
        num_timesteps = 1

        def __call__(self, **kwargs):  # noqa: ANN003
            return SimpleNamespace(images=[_dummy_image()])

    real_import = __import__

    def _import_without_torch(name, *args, **kwargs):  # noqa: ANN002, ANN003
        if name == "torch":
            msg = "No module named 'torch'"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_without_torch):
        with pytest.raises(RuntimeError) as exc_info:
            pipeline_runner.run_pipeline(
                pipe=_NoopPipe(),
                process_parameter=_dummy_process_parameter(),
                input_img=_dummy_image(),
                uv_mask=_dummy_image(),
                canny_img=_dummy_image(),
                normal_img=_dummy_image(),
                depth_img=_dummy_image(),
                progress_callback=lambda _percent: None,
                should_cancel=lambda: False,
            )

    assert "Install Python Dependencies" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_run_pipeline_raises_clear_error_on_cuda_oom() -> None:
    pipeline_runner = _load_pipeline_runner()

    class _Cuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def empty_cache() -> None:
            return

    fake_torch = ModuleType("torch")
    fake_torch.cuda = _Cuda

    class _OomPipe:
        num_timesteps = 1

        def __call__(self, **kwargs):  # noqa: ANN003
            msg = "CUDA out of memory"
            raise fake_torch.cuda.OutOfMemoryError(msg)

    with patch.dict(sys.modules, {"torch": fake_torch}):
        with pytest.raises(RuntimeError) as exc_info:
            pipeline_runner.run_pipeline(
                pipe=_OomPipe(),
                process_parameter=_dummy_process_parameter(),
                input_img=_dummy_image(),
                uv_mask=_dummy_image(),
                canny_img=_dummy_image(),
                normal_img=_dummy_image(),
                depth_img=_dummy_image(),
                progress_callback=lambda _percent: None,
                should_cancel=lambda: False,
            )

    assert "out of GPU memory" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, fake_torch.cuda.OutOfMemoryError)


def test_run_pipeline_wraps_runtime_failures_with_context() -> None:
    pipeline_runner = _load_pipeline_runner()

    class _Cuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def empty_cache() -> None:
            return

    fake_torch = ModuleType("torch")
    fake_torch.cuda = _Cuda

    class _FailingPipe:
        num_timesteps = 1

        def __call__(self, **kwargs):  # noqa: ANN003
            msg = "bad prompt"
            raise ValueError(msg)

    with patch.dict(sys.modules, {"torch": fake_torch}):
        with pytest.raises(RuntimeError) as exc_info:
            pipeline_runner.run_pipeline(
                pipe=_FailingPipe(),
                process_parameter=_dummy_process_parameter(),
                input_img=_dummy_image(),
                uv_mask=_dummy_image(),
                canny_img=_dummy_image(),
                normal_img=_dummy_image(),
                depth_img=_dummy_image(),
                progress_callback=lambda _percent: None,
                should_cancel=lambda: False,
            )

    assert "ValueError: bad prompt" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_run_pipeline_returns_image_on_success() -> None:
    pipeline_runner = _load_pipeline_runner()

    class _Cuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def empty_cache() -> None:
            return

    fake_torch = ModuleType("torch")
    fake_torch.cuda = _Cuda

    expected = _dummy_image()

    class _SuccessPipe:
        num_timesteps = 1

        def __call__(self, **kwargs):  # noqa: ANN003
            return SimpleNamespace(images=[expected])

    with patch.dict(sys.modules, {"torch": fake_torch}):
        result = pipeline_runner.run_pipeline(
            pipe=_SuccessPipe(),
            process_parameter=_dummy_process_parameter(),
            input_img=_dummy_image(),
            uv_mask=_dummy_image(),
            canny_img=_dummy_image(),
            normal_img=_dummy_image(),
            depth_img=_dummy_image(),
            progress_callback=lambda _percent: None,
            should_cancel=lambda: False,
        )

    assert isinstance(result, Image.Image)
    assert result is expected


def test_run_pipeline_interrupts_when_cancel_requested() -> None:
    pipeline_runner = _load_pipeline_runner()

    class _Cuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def empty_cache() -> None:
            return

        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = ModuleType("torch")
    fake_torch.cuda = _Cuda

    class _InterruptiblePipe:
        num_timesteps = 4
        _interrupt = False

        def __call__(self, **kwargs):  # noqa: ANN003
            callback = kwargs["callback_on_step_end"]
            callback(self, 0, 0, {})
            return SimpleNamespace(images=[_dummy_image()])

    with patch.dict(sys.modules, {"torch": fake_torch}):
        with pytest.raises(pipeline_runner.TextureGenerationCancelledError) as exc_info:
            pipeline_runner.run_pipeline(
                pipe=_InterruptiblePipe(),
                process_parameter=_dummy_process_parameter(),
                input_img=_dummy_image(),
                uv_mask=_dummy_image(),
                canny_img=_dummy_image(),
                normal_img=_dummy_image(),
                depth_img=_dummy_image(),
                progress_callback=lambda _percent: None,
                should_cancel=lambda: True,
            )

    assert str(exc_info.value) == "Cancelled by user."
