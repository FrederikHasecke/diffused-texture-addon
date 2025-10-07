import os

import pytest
import diffusers

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLControlNetUnionInpaintPipeline,
    ControlNetModel,
    ControlNetUnionModel,
)


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(
            {
                "pipe_cls": StableDiffusionControlNetInpaintPipeline,
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "cuda",
                "controlnet": ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth",
                ),
            },
            id="runwayml/stable-diffusion-v1-5",
        ),
        pytest.param(
            {
                "pipe_cls": StableDiffusionXLControlNetUnionInpaintPipeline,
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "device": "cuda",
                "controlnet": ControlNetUnionModel.from_pretrained(
                    "xinsir/controlnet-union-sdxl-1.0",
                ),
            },
            id="stabilityai/stable-diffusion-xl-base-1.0",
        ),
    ],
)
def test_create_pipeline_from_pretrained(cfg: dict):
    pipe = cfg["pipe_cls"].from_pretrained(
        cfg["model_id"],
        controlnet=cfg["controlnet"],
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(cfg["device"])
