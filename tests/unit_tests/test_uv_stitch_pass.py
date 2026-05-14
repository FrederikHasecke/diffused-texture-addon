# ruff: noqa: ANN202, ARG005, S101

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np


def _load_uv_stitch_pass_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "uv_stitch_pass.py"
    module_name = f"diffused_texture_addon.diffusedtexture.uv_stitch_pass_{uuid4().hex}"

    addon_pkg = ModuleType("diffused_texture_addon")
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType("diffused_texture_addon.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    blender_ops = ModuleType("diffused_texture_addon.blender_operations")

    @dataclass
    class _UVPassAssets:
        normal_map: np.ndarray
        position_map: np.ndarray
        uv_layout: np.ndarray
        surface_mask: np.ndarray
        seam_link_source_yx: np.ndarray | None = None
        seam_link_target_yx: np.ndarray | None = None

    blender_ops.ProcessParameter = object
    blender_ops.UVPassAssets = _UVPassAssets

    model_support = ModuleType("diffused_texture_addon.model_support")
    model_support.get_default_sd_resolution = (
        lambda sd_version, custom_sd_resolution: custom_sd_resolution or 64
    )

    process_operations = ModuleType(
        "diffused_texture_addon.diffusedtexture.process_operations"
    )
    process_operations._require_cv2 = lambda: None
    process_operations.smooth_alpha_map = lambda alpha_map: cv2.GaussianBlur(
        np.clip(alpha_map.astype(np.float32), 0.0, 1.0),
        (3, 3),
        0,
    )
    process_operations.blend_texture_with_alpha = (
        lambda base_texture, overlay_texture, alpha_map: np.clip(
            np.rint(
                (overlay_texture.astype(np.float32) * alpha_map[..., None])
                + (base_texture.astype(np.float32) * (1.0 - alpha_map[..., None])),
            ),
            0,
            255,
        ).astype(np.uint8)
    )

    pipeline_pkg = ModuleType("diffused_texture_addon.diffusedtexture.pipeline")
    pipeline_builder = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.pipeline_builder"
    )
    pipeline_builder.create_diffusion_pipeline = lambda process_parameter: None
    pipeline_runner = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.pipeline_runner"
    )
    pipeline_runner.run_pipeline = lambda **kwargs: None
    pipeline_runner.TextureGenerationCancelledError = RuntimeError

    with patch.dict(
        sys.modules,
        {
            "diffused_texture_addon": addon_pkg,
            "diffused_texture_addon.blender_operations": blender_ops,
            "diffused_texture_addon.model_support": model_support,
            "diffused_texture_addon.diffusedtexture": diffusedtexture_pkg,
            "diffused_texture_addon.diffusedtexture.process_operations": (
                process_operations
            ),
            "diffused_texture_addon.diffusedtexture.pipeline": pipeline_pkg,
            "diffused_texture_addon.diffusedtexture.pipeline.pipeline_builder": (
                pipeline_builder
            ),
            "diffused_texture_addon.diffusedtexture.pipeline.pipeline_runner": (
                pipeline_runner
            ),
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load uv_stitch_pass module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _build_uv_assets(module: ModuleType, *, linked: bool = True):
    seam_link_source_yx = np.array([[3, 2], [3, 6]], dtype=np.int32) if linked else None
    seam_link_target_yx = np.array([[3, 6], [3, 2]], dtype=np.int32) if linked else None
    return module.UVPassAssets(
        normal_map=np.zeros((8, 8, 4), dtype=np.float32),
        position_map=np.zeros((8, 8, 4), dtype=np.float32),
        uv_layout=np.zeros((8, 8, 4), dtype=np.float32),
        surface_mask=np.ones((8, 8), dtype=np.uint8) * 255,
        seam_link_source_yx=seam_link_source_yx,
        seam_link_target_yx=seam_link_target_yx,
    )


def test_uv_stitch_pass_reduces_topology_seam_delta_only_locally() -> None:
    uv_stitch_pass = _load_uv_stitch_pass_module()
    texture = np.full((8, 8, 3), 128, dtype=np.uint8)
    texture[3, 2] = np.array([255, 0, 0], dtype=np.uint8)
    texture[3, 6] = np.array([0, 0, 255], dtype=np.uint8)
    uv_assets = _build_uv_assets(uv_stitch_pass)

    stitched = uv_stitch_pass.uv_stitch_pass(texture, uv_assets)

    assert np.linalg.norm(
        stitched[3, 2].astype(np.float32) - stitched[3, 6].astype(np.float32)
    ) < np.linalg.norm(
        texture[3, 2].astype(np.float32) - texture[3, 6].astype(np.float32)
    )
    assert np.array_equal(stitched[0, 0], texture[0, 0])


def test_uv_stitch_pass_leaves_texture_unchanged_without_topology_links() -> None:
    uv_stitch_pass = _load_uv_stitch_pass_module()
    texture = np.full((8, 8, 3), 128, dtype=np.uint8)
    texture[3, 2] = np.array([255, 0, 0], dtype=np.uint8)
    uv_assets = _build_uv_assets(uv_stitch_pass, linked=False)

    stitched = uv_stitch_pass.uv_stitch_pass(texture, uv_assets)

    assert np.array_equal(stitched, texture)


def test_accumulate_stitch_confidence_counts_uv_origin_texels() -> None:
    uv_stitch_pass = _load_uv_stitch_pass_module()
    uv_image = np.zeros((2, 2, 4), dtype=np.float32)
    uv_image[0, 0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    facing_image = np.zeros((2, 2), dtype=np.float32)
    facing_image[0, 0] = 0.5

    confidence = uv_stitch_pass.accumulate_stitch_confidence(
        {"uv": [uv_image], "facing": [facing_image]},
        texture_resolution=4,
    )

    assert confidence[3, 0] > 0.5
    assert np.count_nonzero(confidence) == 1
