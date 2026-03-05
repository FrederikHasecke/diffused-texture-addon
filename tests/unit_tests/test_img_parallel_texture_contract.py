import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PIL import Image


def _load_img_parallel_module(captured: dict[str, np.ndarray]) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "img_parallel.py"
    module_name = f"diffused_texture_addon.diffusedtexture.img_parallel_{uuid4().hex}"

    addon_pkg = ModuleType("diffused_texture_addon")
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType("diffused_texture_addon.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    pipeline_pkg = ModuleType("diffused_texture_addon.diffusedtexture.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "diffusedtexture" / "pipeline")]

    blender_ops = ModuleType("diffused_texture_addon.blender_operations")
    blender_ops.ProcessParameter = object

    process_ops = ModuleType(
        "diffused_texture_addon.diffusedtexture.process_operations"
    )

    def assemble_multiview_grid(  # noqa: ANN202
        texture: np.ndarray | None,
        multiview_images: dict[str, list[np.ndarray]],
        render_resolution: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        del multiview_images, render_resolution
        captured["texture"] = texture

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        resized = {
            "input_grid": rgb,
            "content_grid": np.zeros((2, 2), dtype=np.uint8),
            "canny_grid": rgb,
            "normal_grid": rgb,
            "depth_grid": rgb,
        }
        return {}, resized

    def process_uv_texture(**kwargs):  # noqa: ANN003, ANN202
        del kwargs
        return np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)

    process_ops.assemble_multiview_grid = assemble_multiview_grid
    process_ops.process_uv_texture = process_uv_texture

    pipeline_builder = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.pipeline_builder",
    )
    pipeline_builder.create_diffusion_pipeline = lambda process_parameter: object()

    pipeline_runner = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.pipeline_runner",
    )
    pipeline_runner.run_pipeline = lambda **kwargs: Image.fromarray(
        np.zeros((2, 2, 3), dtype=np.uint8),
    )

    with patch.dict(
        sys.modules,
        {
            "diffused_texture_addon": addon_pkg,
            "diffused_texture_addon.diffusedtexture": diffusedtexture_pkg,
            "diffused_texture_addon.diffusedtexture.pipeline": pipeline_pkg,
            "diffused_texture_addon.blender_operations": blender_ops,
            "diffused_texture_addon.diffusedtexture.process_operations": process_ops,
            "diffused_texture_addon.diffusedtexture.pipeline.pipeline_builder": pipeline_builder,
            "diffused_texture_addon.diffusedtexture.pipeline.pipeline_runner": pipeline_runner,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load img_parallel module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_img_parallel_keeps_texture_in_normalized_float_space() -> None:
    captured: dict[str, np.ndarray] = {}
    img_parallel_module = _load_img_parallel_module(captured)

    texture = np.array(
        [
            [[0.05, 0.5, 0.95], [0.2, 0.4, 0.6]],
            [[0.1, 0.3, 0.7], [0.8, 0.25, 0.9]],
        ],
        dtype=np.float32,
    )

    process_parameter = SimpleNamespace(
        render_resolution=2,
        texture_resolution=4,
        denoise_strength=1.0,
        guidance_scale=7.5,
        num_inference_steps=1,
    )
    multiview_images = {
        "uv": [np.zeros((2, 2, 3), dtype=np.float32)],
        "facing": [np.zeros((2, 2, 1), dtype=np.float32)],
    }

    img_parallel_module.img_parallel(
        multiview_images=multiview_images,
        process_parameter=process_parameter,
        progress_callback=lambda _progress: None,
        texture=texture,
    )

    assert "texture" in captured
    projected_input = captured["texture"]
    assert projected_input.dtype == np.float32
    assert np.allclose(projected_input, texture)
