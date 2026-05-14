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


def _load_uv_repair_pass_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "uv_repair_pass.py"
    module_name = f"diffused_texture_addon.diffusedtexture.uv_repair_pass_{uuid4().hex}"

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
        seam_line_mask: np.ndarray | None = None
        seam_link_source_yx: np.ndarray | None = None
        seam_link_target_yx: np.ndarray | None = None
        seam_unresolved_link_mask: np.ndarray | None = None

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
            msg = "Could not load uv_repair_pass module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_couple_mask_across_topology_links_marks_exact_partner_only() -> None:
    uv_repair_pass = _load_uv_repair_pass_module()
    repair_mask = np.zeros((8, 8), dtype=np.uint8)
    repair_mask[2, 2] = 255

    coupled = uv_repair_pass._couple_mask_across_topology_links(
        repair_mask,
        np.array([[2, 2], [5, 5]], dtype=np.int32),
        np.array([[2, 6], [5, 6]], dtype=np.int32),
    )

    assert coupled[2, 2] == 255
    assert coupled[2, 6] == 255
    assert coupled[5, 5] == 0
    assert coupled[5, 6] == 0


def test_topology_seam_mismatch_mask_is_pairwise() -> None:
    uv_repair_pass = _load_uv_repair_pass_module()
    texture = np.full((8, 8, 3), 128, dtype=np.uint8)
    texture[2, 2] = np.array([255, 0, 0], dtype=np.uint8)
    texture[2, 6] = np.array([0, 0, 255], dtype=np.uint8)
    texture[5, 5] = np.array([64, 64, 64], dtype=np.uint8)
    texture[5, 6] = np.array([65, 65, 65], dtype=np.uint8)

    mismatch = uv_repair_pass._build_topology_seam_mismatch_mask(
        texture,
        np.array([[2, 2], [5, 5]], dtype=np.int32),
        np.array([[2, 6], [5, 6]], dtype=np.int32),
        np.full((8, 8), 255, dtype=np.uint8),
    )

    assert mismatch[2, 2] == 255
    assert mismatch[2, 6] == 255
    assert mismatch[5, 5] == 0
    assert mismatch[5, 6] == 0


def test_accumulate_uv_repair_diagnostics_counts_uv_origin_texels() -> None:
    uv_repair_pass = _load_uv_repair_pass_module()
    uv_image = np.zeros((2, 2, 4), dtype=np.float32)
    uv_image[0, 0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    facing_image = np.zeros((2, 2), dtype=np.float32)
    facing_image[0, 0] = 0.25

    coverage_count, best_facing, weight_sum = (
        uv_repair_pass.accumulate_uv_repair_diagnostics(
            {"uv": [uv_image], "facing": [facing_image]},
            texture_resolution=4,
        )
    )

    assert coverage_count[3, 0] == 1.0
    assert best_facing[3, 0] == 0.25
    assert weight_sum[3, 0] == 0.25
    assert np.count_nonzero(coverage_count) == 1
