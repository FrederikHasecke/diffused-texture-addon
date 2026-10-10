from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage


class _DummyImage:
    def __init__(self) -> None:
        self.size = (2, 1)
        self.channels = 4
        self.pixels = [
            0.1,
            0.2,
            0.3,
            1.0,
            0.4,
            0.5,
            0.6,
            1.0,
        ]


def test_extract_process_parameter_snapshots_thread_safe_data() -> None:
    # import through the package so that relative imports inside
    # `blender_operations` resolve correctly when running under pytest
    from diffused_texture_addon.blender_operations import (
        extract_process_parameter_from_context,
    )

    scene = SimpleNamespace(
        num_loras=1,
        lora_models=[SimpleNamespace(path="style.safetensors", strength=0.75)],
        use_ipadapter=True,
        ipadapter_image=_DummyImage(),
    )
    context = SimpleNamespace(scene=scene)

    process_parameter = extract_process_parameter_from_context(context)

    assert process_parameter.lora_models == [
        {
            "path": "style.safetensors",
            "strength": 0.75,
        },
    ]
    assert process_parameter.lora_models is not scene.lora_models

    scene.lora_models[0].path = "changed.safetensors"
    assert process_parameter.lora_models[0]["path"] == "style.safetensors"

    assert isinstance(process_parameter.ipadapter_image, np.ndarray)
    assert process_parameter.ipadapter_image.shape == (1, 2, 4)


def test_load_img_to_numpy_releases_loaded_image(tmp_path: Path) -> None:
    # same reasoning as above; use package import
    import bpy
    from diffused_texture_addon.blender_operations import load_img_to_numpy

    file_name = f"tmp_image_{uuid4().hex}.png"
    image_path = tmp_path / file_name

    image_data = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8)
    PILImage.fromarray(image_data, mode="RGBA").save(image_path)

    initial_count = len(bpy.data.images)
    loaded = load_img_to_numpy(image_path)

    assert loaded.shape == (1, 2, 4)
    assert loaded.dtype == np.float32
    assert len(bpy.data.images) == initial_count


def test_extract_process_parameter_resolves_blend_relative_model_paths(
    tmp_path: Path,
) -> None:
    import bpy

    from diffused_texture_addon.blender_operations import (
        extract_process_parameter_from_context,
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / "project.blend"))

    scene = SimpleNamespace(
        num_loras=1,
        lora_models=[SimpleNamespace(path="//loras/style.safetensors", strength=1.0)],
        use_ipadapter=False,
        checkpoint_path="//models/base.safetensors",
        controlnet_union_path="xinsir/controlnet-union-sdxl-1.0",
        depth_controlnet_path="//controlnets/depth",
        canny_controlnet_path="/abs/canny",
        normal_controlnet_path="",
    )

    params = extract_process_parameter_from_context(SimpleNamespace(scene=scene))

    assert Path(params.checkpoint_path) == tmp_path / "models" / "base.safetensors"
    assert Path(params.lora_models[0]["path"]) == tmp_path / "loras" / "style.safetensors"
    assert Path(params.depth_controlnet_path) == tmp_path / "controlnets" / "depth"
    # Hub ids, absolute paths and empty values pass through untouched.
    assert params.controlnet_union_path == "xinsir/controlnet-union-sdxl-1.0"
    assert params.canny_controlnet_path == "/abs/canny"
    assert params.normal_controlnet_path == ""
