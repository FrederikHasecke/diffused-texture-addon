import os

import pytest

from dotenv import load_dotenv

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.network,
]



def test_create_pipeline_sd15_inpaint():
    """Test creating StableDiffusion 1.5 Inpaint pipeline with ControlNet."""
    try:
        import torch
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline,
            ControlNetModel,
        )
        # Set local hf cache path
        load_dotenv()
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            os.environ["HF_HOME"] = hf_home
        

    except ImportError:
        pytest.fail("Required dependencies not available")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    assert pipe is not None


def test_create_pipeline_sdxl_inpaint():
    """Test creating SDXL Inpaint pipeline with ControlNet Union."""
    try:
        import torch
        from diffusers import (
            StableDiffusionXLControlNetUnionInpaintPipeline,
            ControlNetUnionModel,
        )
        # Set local hf cache path
        load_dotenv()
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            os.environ["HF_HOME"] = hf_home
    except ImportError:
        pytest.fail("Required dependencies not available")

    controlnet = ControlNetUnionModel.from_pretrained(
        "xinsir/controlnet-union-sdxl-1.0",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    assert pipe is not None
