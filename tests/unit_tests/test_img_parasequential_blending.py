import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PIL import Image


def _blend_texture_with_alpha(
    base_texture: np.ndarray,
    overlay_texture: np.ndarray,
    alpha_map: np.ndarray,
) -> np.ndarray:
    alpha = alpha_map.astype(np.float32)[..., None]
    blended = overlay_texture.astype(np.float32) * alpha + base_texture.astype(
        np.float32
    ) * (1.0 - alpha)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _load_img_parasequential_module(
    process_uv_texture_impl,
    assemble_multiview_subgrid_impl,
) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "img_parasequential.py"
    package_name = f"diffused_texture_addon_{uuid4().hex}"
    module_name = f"{package_name}.diffusedtexture.img_parasequential_{uuid4().hex}"

    addon_pkg = ModuleType(package_name)
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType(f"{package_name}.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    pipeline_pkg = ModuleType(f"{package_name}.diffusedtexture.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "diffusedtexture" / "pipeline")]

    blender_ops = ModuleType(f"{package_name}.blender_operations")
    blender_ops.ProcessParameter = object

    model_support = ModuleType(f"{package_name}.model_support")
    model_support.get_default_sd_resolution = lambda *_args, **_kwargs: 2

    process_ops = ModuleType(f"{package_name}.diffusedtexture.process_operations")
    process_ops._require_cv2 = lambda: None
    process_ops.assemble_multiview_subgrid = assemble_multiview_subgrid_impl
    process_ops.blend_texture_with_alpha = _blend_texture_with_alpha
    process_ops.inpaint_missing = (
        lambda target_resolution, all_texture_array, all_texture_weight_array: np.zeros(
            (target_resolution, target_resolution, 3),
            dtype=np.uint8,
        )
    )
    process_ops.process_uv_texture = process_uv_texture_impl
    process_ops.smooth_alpha_map = lambda alpha: alpha.astype(np.float32)

    pipeline_builder = ModuleType(
        f"{package_name}.diffusedtexture.pipeline.pipeline_builder",
    )
    pipeline_builder.create_diffusion_pipeline = lambda _process_parameter: object()

    pipeline_runner = ModuleType(
        f"{package_name}.diffusedtexture.pipeline.pipeline_runner",
    )
    pipeline_runner.run_pipeline = lambda **_kwargs: Image.fromarray(
        np.zeros((2, 2, 3), dtype=np.uint8),
    )

    with patch.dict(
        sys.modules,
        {
            package_name: addon_pkg,
            f"{package_name}.diffusedtexture": diffusedtexture_pkg,
            f"{package_name}.diffusedtexture.pipeline": pipeline_pkg,
            f"{package_name}.blender_operations": blender_ops,
            f"{package_name}.model_support": model_support,
            f"{package_name}.diffusedtexture.process_operations": process_ops,
            f"{package_name}.diffusedtexture.pipeline.pipeline_builder": pipeline_builder,
            f"{package_name}.diffusedtexture.pipeline.pipeline_runner": pipeline_runner,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load img_parasequential module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_img_parasequential_forwards_facing_percentile_to_process_uv_texture() -> None:
    captured: dict[str, float] = {}

    def assemble_multiview_subgrid_impl(**_kwargs):  # noqa: ANN003, ANN202
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        resized = {
            "input_grid": rgb,
            "content_grid": np.zeros((2, 2), dtype=np.uint8),
            "canny_grid": rgb,
            "normal_grid": rgb,
            "depth_grid": rgb,
        }
        return [], [resized]

    def process_uv_texture_impl(**kwargs):  # noqa: ANN003, ANN202
        captured["facing_percentile"] = kwargs["facing_percentile"]
        return np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)

    img_parasequential_module = _load_img_parasequential_module(
        process_uv_texture_impl,
        assemble_multiview_subgrid_impl,
    )

    process_parameter = SimpleNamespace(
        sd_version="sd15",
        custom_sd_resolution=None,
        render_resolution=2,
        texture_resolution=4,
        num_cameras=1,
        denoise_strength=1.0,
        guidance_scale=7.5,
        num_inference_steps=1,
    )
    multiview_images = {
        "depth": [np.zeros((2, 2, 3), dtype=np.float32)],
        "normal": [np.zeros((2, 2, 3), dtype=np.float32)],
        "facing": [np.ones((2, 2, 1), dtype=np.float32)],
        "uv": [np.zeros((2, 2, 3), dtype=np.float32)],
    }

    img_parasequential_module.img_parasequential(
        multiview_images=multiview_images,
        process_parameter=process_parameter,
        progress_callback=lambda _progress: None,
        facing_percentile=0.3,
        subgrid_rows=1,
        subgrid_cols=1,
    )

    assert captured["facing_percentile"] == 0.3


def test_img_parasequential_soft_blends_between_subgrids() -> None:
    texture_inputs: list[np.ndarray] = []
    call_count = {"process_uv": 0}

    def assemble_multiview_subgrid_impl(**kwargs):  # noqa: ANN003, ANN202
        texture_inputs.append(kwargs["texture"].copy())
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        resized = {
            "input_grid": rgb,
            "content_grid": np.zeros((2, 2), dtype=np.uint8),
            "canny_grid": rgb,
            "normal_grid": rgb,
            "depth_grid": rgb,
        }
        return [], [resized]

    def process_uv_texture_impl(**kwargs):  # noqa: ANN003, ANN202
        del kwargs
        call_count["process_uv"] += 1
        if call_count["process_uv"] == 1:
            return (
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.full((4, 4), 128, dtype=np.uint8),
            )

        return np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)

    img_parasequential_module = _load_img_parasequential_module(
        process_uv_texture_impl,
        assemble_multiview_subgrid_impl,
    )

    process_parameter = SimpleNamespace(
        sd_version="sd15",
        custom_sd_resolution=None,
        render_resolution=2,
        texture_resolution=4,
        num_cameras=2,
        denoise_strength=1.0,
        guidance_scale=7.5,
        num_inference_steps=1,
    )
    multiview_images = {
        "depth": [
            np.zeros((2, 2, 3), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.float32),
        ],
        "normal": [
            np.zeros((2, 2, 3), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.float32),
        ],
        "facing": [
            np.ones((2, 2, 1), dtype=np.float32),
            np.ones((2, 2, 1), dtype=np.float32),
        ],
        "uv": [
            np.zeros((2, 2, 3), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.float32),
        ],
    }

    img_parasequential_module.img_parasequential(
        multiview_images=multiview_images,
        process_parameter=process_parameter,
        progress_callback=lambda _progress: None,
        facing_percentile=0.5,
        subgrid_rows=1,
        subgrid_cols=1,
    )

    assert len(texture_inputs) == 2
    assert np.all(texture_inputs[1] > 0.4)
    assert np.all(texture_inputs[1] < 0.6)
