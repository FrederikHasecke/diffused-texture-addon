# ruff: noqa: ANN202, ARG005, S101, TC006

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest


def _load_texture_generation_module(calls: dict[str, object]) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "texture_generation.py"
    module_name = f"diffused_texture_addon.texture_generation_{uuid4().hex}"

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
    blender_ops.load_img_to_numpy = lambda path: np.zeros((1, 1, 4), dtype=np.float32)

    img_parallel = ModuleType("diffused_texture_addon.diffusedtexture.img_parallel")
    img_parallel.img_parallel = lambda **kwargs: calls.setdefault(
        "img_parallel",
        np.full((4, 4, 3), 64, dtype=np.uint8),
    )

    img_sequential = ModuleType("diffused_texture_addon.diffusedtexture.img_sequential")
    img_sequential.img_sequential = lambda **kwargs: np.full(
        (4, 4, 3),
        96,
        dtype=np.uint8,
    )

    img_parasequential = ModuleType(
        "diffused_texture_addon.diffusedtexture.img_parasequential",
    )
    img_parasequential.img_parasequential = lambda **kwargs: np.full(
        (4, 4, 3),
        128,
        dtype=np.uint8,
    )

    uv_stitch_pass = ModuleType("diffused_texture_addon.diffusedtexture.uv_stitch_pass")

    def _uv_stitch_pass(**kwargs) -> np.ndarray:  # noqa: ANN003
        calls["uv_stitch_args"] = kwargs
        return np.full((4, 4, 3), 192, dtype=np.uint8)

    uv_stitch_pass.uv_stitch_pass = _uv_stitch_pass

    with patch.dict(
        sys.modules,
        {
            "diffused_texture_addon": addon_pkg,
            "diffused_texture_addon.blender_operations": blender_ops,
            "diffused_texture_addon.diffusedtexture": diffusedtexture_pkg,
            "diffused_texture_addon.diffusedtexture.img_parallel": img_parallel,
            "diffused_texture_addon.diffusedtexture.img_sequential": img_sequential,
            "diffused_texture_addon.diffusedtexture.img_parasequential": (
                img_parasequential
            ),
            "diffused_texture_addon.diffusedtexture.uv_stitch_pass": uv_stitch_pass,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load texture_generation module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _build_uv_assets(module: ModuleType):
    return module.UVPassAssets(
        normal_map=np.zeros((4, 4, 4), dtype=np.float32),
        position_map=np.zeros((4, 4, 4), dtype=np.float32),
        uv_layout=np.zeros((4, 4, 4), dtype=np.float32),
        surface_mask=np.ones((4, 4), dtype=np.uint8) * 255,
    )


def test_run_texture_generation_runs_auto_uv_stitch_for_image_modes() -> None:
    calls: dict[str, object] = {}
    texture_generation = _load_texture_generation_module(calls)
    uv_assets = _build_uv_assets(texture_generation)
    return_bucket: list[np.ndarray] = []
    done_calls: list[bool] = []

    texture_generation.run_texture_generation(
        process_parameter=SimpleNamespace(
            operation_mode="PARALLEL_IMG",
            subgrid_rows=2,
            subgrid_cols=2,
        ),
        generation_inputs={"uv": [], "facing": []},
        progress_callback=lambda _value: None,
        should_cancel=lambda: False,
        mark_done=lambda success=True: done_calls.append(success),
        return_texture_bucket=return_bucket,
        texture=None,
        uv_assets=uv_assets,
    )

    assert "uv_stitch_args" in calls
    repair_args = cast(dict[str, Any], calls["uv_stitch_args"])
    base_texture = cast(np.ndarray, calls["img_parallel"])
    assert np.array_equal(repair_args["texture"], base_texture)
    assert repair_args["uv_assets"] is uv_assets
    assert np.array_equal(return_bucket[0], np.full((4, 4, 3), 192, dtype=np.uint8))
    assert done_calls == [True]


def test_run_texture_generation_skips_auto_uv_stitch_without_uv_assets() -> None:
    calls: dict[str, object] = {}
    texture_generation = _load_texture_generation_module(calls)
    return_bucket: list[np.ndarray] = []

    texture_generation.run_texture_generation(
        process_parameter=SimpleNamespace(
            operation_mode="PARALLEL_IMG",
            subgrid_rows=2,
            subgrid_cols=2,
        ),
        generation_inputs={"uv": [], "facing": []},
        progress_callback=lambda _value: None,
        should_cancel=lambda: False,
        mark_done=lambda success=True: None,
        return_texture_bucket=return_bucket,
        texture=None,
    )

    assert "uv_stitch_args" not in calls
    assert np.array_equal(return_bucket[0], np.full((4, 4, 3), 64, dtype=np.uint8))


def test_run_texture_generation_rejects_disabled_uv_mode() -> None:
    calls: dict[str, object] = {}
    texture_generation = _load_texture_generation_module(calls)
    uv_assets = _build_uv_assets(texture_generation)
    return_bucket: list[np.ndarray] = []

    with pytest.raises(ValueError, match="UV_PASS"):
        texture_generation.run_texture_generation(
            process_parameter=SimpleNamespace(operation_mode="UV_PASS"),
            generation_inputs={"uv": [], "facing": []},
            progress_callback=lambda _value: None,
            should_cancel=lambda: False,
            mark_done=lambda success=True: None,
            return_texture_bucket=return_bucket,
            texture=None,
            uv_assets=uv_assets,
        )

    assert "uv_stitch_args" not in calls
    assert return_bucket == []
