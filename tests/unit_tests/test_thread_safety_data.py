from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

bpy = pytest.importorskip("bpy")
blender_operations = pytest.importorskip(
    "diffused_texture_addon.blender_operations",
)

extract_process_parameter_from_context = (
    blender_operations.extract_process_parameter_from_context
)
load_img_to_numpy = blender_operations.load_img_to_numpy


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
    file_name = f"tmp_image_{uuid4().hex}.png"
    image_path = tmp_path / file_name

    image_data = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8)
    PILImage.fromarray(image_data, mode="RGBA").save(image_path)

    initial_count = len(bpy.data.images)
    loaded = load_img_to_numpy(image_path)

    assert loaded.shape == (1, 2, 4)
    assert loaded.dtype == np.float32
    assert len(bpy.data.images) == initial_count
