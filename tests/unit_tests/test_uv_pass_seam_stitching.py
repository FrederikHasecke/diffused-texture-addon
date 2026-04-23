import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np


def _load_uv_pass_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "uv_pass.py"
    module_name = f"diffused_texture_addon.diffusedtexture.uv_pass_{uuid4().hex}"

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

    blender_ops.ProcessParameter = object
    blender_ops.UVPassAssets = _UVPassAssets

    model_support = ModuleType("diffused_texture_addon.model_support")
    model_support.get_default_sd_resolution = (
        lambda sd_version, custom_sd_resolution: custom_sd_resolution or 64
    )

    process_operations = ModuleType(
        "diffused_texture_addon.diffusedtexture.process_operations"
    )

    def _require_cv2() -> None:
        return

    def _smooth_alpha_map(alpha_map: np.ndarray) -> np.ndarray:
        alpha_map = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)
        return np.clip(cv2.GaussianBlur(alpha_map, (3, 3), 0), 0.0, 1.0)

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

    process_operations._require_cv2 = _require_cv2
    process_operations.smooth_alpha_map = _smooth_alpha_map
    process_operations.blend_texture_with_alpha = _blend_texture_with_alpha

    pipeline_builder = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.pipeline_builder"
    )
    pipeline_builder.create_diffusion_pipeline = lambda process_parameter: None

    pipeline_runner = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.pipeline_runner"
    )
    pipeline_runner.run_pipeline = lambda **kwargs: None

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
            msg = "Could not load uv_pass module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _synthetic_surface(
    texture_resolution: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_map = np.zeros(
        (texture_resolution, texture_resolution, 4), dtype=np.float32
    )
    normal_map = np.zeros((texture_resolution, texture_resolution, 4), dtype=np.float32)
    uv_layout = np.ones((texture_resolution, texture_resolution, 4), dtype=np.float32)
    surface_mask = np.full(
        (texture_resolution, texture_resolution), 255, dtype=np.uint8
    )

    for y in range(texture_resolution):
        for x in range(texture_resolution):
            position_map[y, x, :3] = np.array(
                [x * 1e-4, y * 1e-4, 0.25],
                dtype=np.float32,
            )
            normal_map[y, x, :3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            normal_map[y, x, 3] = 1.0

    return position_map, normal_map, uv_layout, surface_mask


def _rotation_matrix_y(angle_radians: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle_radians), 0.0, np.sin(angle_radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_radians), 0.0, np.cos(angle_radians)],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_z(angle_radians: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0.0],
            [np.sin(angle_radians), np.cos(angle_radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _synthetic_curved_chart(
    texture_resolution: int = 64,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(-1.0, 1.0, texture_resolution, dtype=np.float32)
    u_coords, v_coords = np.meshgrid(grid, grid)

    height = (0.35 * (u_coords**2)) + (0.2 * (v_coords**2))
    position = np.stack((u_coords, v_coords, height), axis=-1)

    du = np.stack(
        (np.ones_like(u_coords), np.zeros_like(u_coords), 0.7 * u_coords),
        axis=-1,
    )
    dv = np.stack(
        (np.zeros_like(v_coords), np.ones_like(v_coords), 0.4 * v_coords),
        axis=-1,
    )
    normals = np.cross(du, dv)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    if rotation is not None:
        position = position @ rotation.T
        normals = normals @ rotation.T

    position_map = np.zeros(
        (texture_resolution, texture_resolution, 4), dtype=np.float32
    )
    position_map[..., :3] = position.astype(np.float32)
    position_map[..., 3] = 1.0

    normal_map = np.zeros((texture_resolution, texture_resolution, 4), dtype=np.float32)
    normal_map[..., :3] = normals.astype(np.float32)
    normal_map[..., 3] = 1.0

    surface_mask = np.full(
        (texture_resolution, texture_resolution), 255, dtype=np.uint8
    )
    return position_map, normal_map, surface_mask


def test_find_position_seam_groups_links_remote_uv_texels() -> None:
    uv_pass = _load_uv_pass_module()
    position_map, normal_map, _uv_layout, surface_mask = _synthetic_surface()

    seam_band = np.zeros_like(surface_mask)
    seam_band[20, 10] = 255
    seam_band[20, 45] = 255
    position_map[20, 10, :3] = np.array([0.4, 0.2, 0.1], dtype=np.float32)
    position_map[20, 45, :3] = np.array([0.4001, 0.2, 0.1], dtype=np.float32)

    seam_groups = uv_pass.find_position_seam_groups(
        position_map=position_map,
        normal_map=normal_map,
        seam_band=seam_band,
        surface_mask=surface_mask,
        texture_resolution=64,
    )

    assert len(seam_groups) == 1
    assert {tuple(coord.tolist()) for coord in seam_groups[0]} == {(20, 10), (20, 45)}


def test_find_position_seam_groups_rejects_unrelated_surfaces() -> None:
    uv_pass = _load_uv_pass_module()
    position_map, normal_map, _uv_layout, surface_mask = _synthetic_surface()

    seam_band = np.zeros_like(surface_mask)
    seam_band[20, 10] = 255
    seam_band[20, 45] = 255
    position_map[20, 10, :3] = np.array([0.4, 0.2, 0.1], dtype=np.float32)
    position_map[20, 45, :3] = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    normal_map[20, 45, :3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    seam_groups = uv_pass.find_position_seam_groups(
        position_map=position_map,
        normal_map=normal_map,
        seam_band=seam_band,
        surface_mask=surface_mask,
        texture_resolution=64,
    )

    assert seam_groups == []


def test_apply_position_seam_stitching_aligns_group_colors_without_touching_interior() -> (
    None
):
    uv_pass = _load_uv_pass_module()
    position_map, normal_map, uv_layout, surface_mask = _synthetic_surface()
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    texture[20, 10] = np.array([255, 0, 0], dtype=np.uint8)
    texture[20, 45] = np.array([0, 0, 255], dtype=np.uint8)

    position_map[20, 10, :3] = np.array([0.4, 0.2, 0.1], dtype=np.float32)
    position_map[20, 45, :3] = np.array([0.4001, 0.2, 0.1], dtype=np.float32)
    uv_layout[20, 10, 0] = 0.0
    uv_layout[20, 45, 0] = 0.0

    stitched = uv_pass.apply_position_seam_stitching(
        texture=texture,
        position_map=position_map,
        normal_map=normal_map,
        uv_layout=uv_layout,
        surface_mask=surface_mask,
    )

    stitched_mean = np.mean(
        stitched[np.array([20, 20]), np.array([10, 45])].astype(np.float32),
        axis=0,
    )
    original_mean = np.mean(
        texture[np.array([20, 20]), np.array([10, 45])].astype(np.float32),
        axis=0,
    )

    assert np.linalg.norm(
        stitched[20, 10].astype(np.float32) - stitched[20, 45].astype(np.float32)
    ) < np.linalg.norm(
        texture[20, 10].astype(np.float32) - texture[20, 45].astype(np.float32)
    )
    assert np.allclose(stitched_mean, original_mean, atol=8.0)
    assert np.array_equal(stitched[5, 5], texture[5, 5])


def test_apply_position_seam_stitching_matches_texels_adjacent_to_uv_boundary() -> None:
    uv_pass = _load_uv_pass_module()
    position_map, normal_map, uv_layout, surface_mask = _synthetic_surface()
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    texture[21, 10] = np.array([255, 0, 0], dtype=np.uint8)
    texture[21, 45] = np.array([0, 0, 255], dtype=np.uint8)

    position_map[21, 10, :3] = np.array([0.4, 0.2, 0.1], dtype=np.float32)
    position_map[21, 45, :3] = np.array([0.4001, 0.2, 0.1], dtype=np.float32)
    uv_layout[20, 10, 0] = 0.0
    uv_layout[20, 45, 0] = 0.0

    stitched = uv_pass.apply_position_seam_stitching(
        texture=texture,
        position_map=position_map,
        normal_map=normal_map,
        uv_layout=uv_layout,
        surface_mask=surface_mask,
    )

    assert np.linalg.norm(
        stitched[21, 10].astype(np.float32) - stitched[21, 45].astype(np.float32)
    ) < np.linalg.norm(
        texture[21, 10].astype(np.float32) - texture[21, 45].astype(np.float32)
    )
    assert np.array_equal(stitched[5, 5], texture[5, 5])


def test_apply_position_seam_stitching_preserves_local_detail() -> None:
    uv_pass = _load_uv_pass_module()
    position_map, normal_map, uv_layout, surface_mask = _synthetic_surface()
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    checker_offsets = np.array(
        [[0, 24, 0], [24, 0, 24], [0, 24, 0]],
        dtype=np.uint8,
    )

    left_base = np.array([188, 96, 96], dtype=np.uint8)
    right_base = np.array([96, 96, 188], dtype=np.uint8)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            offset = int(checker_offsets[dy + 1, dx + 1])
            texture[20 + dy, 10 + dx] = np.clip(
                left_base.astype(np.int16) + offset,
                0,
                255,
            )
            texture[20 + dy, 45 + dx] = np.clip(
                right_base.astype(np.int16) + offset,
                0,
                255,
            )

    position_map[20, 10, :3] = np.array([0.4, 0.2, 0.1], dtype=np.float32)
    position_map[20, 45, :3] = np.array([0.4001, 0.2, 0.1], dtype=np.float32)
    uv_layout[20, 10, 0] = 0.0
    uv_layout[20, 45, 0] = 0.0

    stitched = uv_pass.apply_position_seam_stitching(
        texture=texture,
        position_map=position_map,
        normal_map=normal_map,
        uv_layout=uv_layout,
        surface_mask=surface_mask,
    )

    before_left_detail = texture[20, 10].astype(np.int16) - texture[20, 11].astype(
        np.int16,
    )
    after_left_detail = stitched[20, 10].astype(np.int16) - stitched[20, 11].astype(
        np.int16,
    )

    assert np.linalg.norm(
        stitched[20, 10].astype(np.float32) - stitched[20, 45].astype(np.float32)
    ) < np.linalg.norm(
        texture[20, 10].astype(np.float32) - texture[20, 45].astype(np.float32)
    )
    assert np.linalg.norm(after_left_detail - before_left_detail) < 20.0
    assert np.array_equal(stitched[5, 5], texture[5, 5])


def test_canonicalize_uv_normal_map_is_invariant_to_global_rotation() -> None:
    uv_pass = _load_uv_pass_module()

    position_map, normal_map, surface_mask = _synthetic_curved_chart()
    rotation = _rotation_matrix_y(0.7) @ _rotation_matrix_z(-0.35)
    rotated_position_map, rotated_normal_map, rotated_surface_mask = (
        _synthetic_curved_chart(
            rotation=rotation,
        )
    )

    canonical = uv_pass.canonicalize_uv_normal_map(
        normal_map=normal_map,
        position_map=position_map,
        surface_mask=surface_mask,
    )
    rotated_canonical = uv_pass.canonicalize_uv_normal_map(
        normal_map=rotated_normal_map,
        position_map=rotated_position_map,
        surface_mask=rotated_surface_mask,
    )

    valid_mask = surface_mask > 0
    assert np.allclose(canonical[valid_mask], rotated_canonical[valid_mask], atol=0.06)
    assert float(np.std(canonical[..., 0][valid_mask])) > 0.02
    assert float(np.std(canonical[..., 1][valid_mask])) > 0.02
