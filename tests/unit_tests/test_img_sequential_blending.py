import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PIL import Image


def _load_img_sequential_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    package_name = f"diffused_texture_addon_{uuid4().hex}"

    addon_pkg = ModuleType(package_name)
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType(f"{package_name}.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    pipeline_pkg = ModuleType(f"{package_name}.diffusedtexture.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "diffusedtexture" / "pipeline")]

    blender_ops = ModuleType(f"{package_name}.blender_operations")
    blender_ops.ProcessParameter = object

    model_support = ModuleType(f"{package_name}.model_support")
    model_support.get_default_sd_resolution = lambda *_args, **_kwargs: 4

    pipeline_builder = ModuleType(
        f"{package_name}.diffusedtexture.pipeline.pipeline_builder",
    )
    pipeline_builder.create_diffusion_pipeline = lambda _process_parameter: object()

    pipeline_runner = ModuleType(
        f"{package_name}.diffusedtexture.pipeline.pipeline_runner",
    )
    pipeline_runner.run_pipeline = lambda **_kwargs: Image.fromarray(
        np.zeros((4, 4, 3), dtype=np.uint8),
    )

    process_ops_path = repo_root / "diffusedtexture" / "process_operations.py"
    process_ops_name = f"{package_name}.diffusedtexture.process_operations"
    process_ops_spec = importlib.util.spec_from_file_location(
        process_ops_name,
        process_ops_path,
    )
    if process_ops_spec is None or process_ops_spec.loader is None:
        msg = "Could not load process_operations module spec."
        raise RuntimeError(msg)

    process_ops_module = importlib.util.module_from_spec(process_ops_spec)

    module_path = repo_root / "diffusedtexture" / "img_sequential.py"
    module_name = f"{package_name}.diffusedtexture.img_sequential_{uuid4().hex}"

    with patch.dict(
        sys.modules,
        {
            package_name: addon_pkg,
            f"{package_name}.diffusedtexture": diffusedtexture_pkg,
            f"{package_name}.diffusedtexture.pipeline": pipeline_pkg,
            f"{package_name}.blender_operations": blender_ops,
            f"{package_name}.model_support": model_support,
            f"{package_name}.diffusedtexture.pipeline.pipeline_builder": pipeline_builder,
            f"{package_name}.diffusedtexture.pipeline.pipeline_runner": pipeline_runner,
        },
    ):
        process_ops_spec.loader.exec_module(process_ops_module)
        process_ops_module._require_cv2 = lambda: None
        sys.modules[process_ops_name] = process_ops_module

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load img_sequential module spec."
            raise RuntimeError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_create_new_view_input_samples_from_full_texture_space() -> None:
    img_sequential_module = _load_img_sequential_module()

    texture = np.zeros((4, 4, 3), dtype=np.uint8)
    texture[3, 0] = [10, 0, 0]
    texture[3, 3] = [20, 0, 0]
    texture[0, 0] = [30, 0, 0]
    texture[0, 3] = [40, 0, 0]

    unpainted_mask = np.zeros((4, 4), dtype=np.uint8)
    unpainted_mask[3, 0] = 1
    unpainted_mask[3, 3] = 2
    unpainted_mask[0, 0] = 3
    unpainted_mask[0, 3] = 4

    input_view = np.zeros((2, 2, 3), dtype=np.uint8)
    uv_view = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    projected_view, projected_mask = img_sequential_module.create_new_view_input(
        texture,
        unpainted_mask,
        input_view,
        uv_view,
    )

    assert np.array_equal(
        np.array(projected_view),
        np.array(
            [
                [[10, 0, 0], [20, 0, 0]],
                [[30, 0, 0], [40, 0, 0]],
            ],
            dtype=np.uint8,
        ),
    )
    assert np.array_equal(
        projected_mask,
        np.array(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=np.uint8,
        ),
    )


def test_project_view_to_texture_soft_blends_existing_texture() -> None:
    img_sequential_module = _load_img_sequential_module()

    sd_result = Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8))
    texture = np.full((4, 4, 3), 100, dtype=np.uint8)
    unpainted_mask = np.full((4, 4), 255, dtype=np.uint8)

    coords = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
    grid_u, grid_v = np.meshgrid(coords, coords)
    uv_view = np.stack((grid_u, grid_v, np.zeros_like(grid_u)), axis=-1)
    facing_view = np.full((4, 4), 128, dtype=np.uint8)

    blended_texture, updated_mask = img_sequential_module.project_view_to_texture(
        sd_result=sd_result,
        uv_view=uv_view,
        facing_view=facing_view,
        texture_resolution=4,
        texture=texture,
        unpainted_mask=unpainted_mask,
        facing_percentile=0.5,
    )

    assert np.all(blended_texture > 100)
    assert np.all(blended_texture < 200)
    assert np.all(updated_mask == 0)
