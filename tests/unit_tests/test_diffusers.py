import os
import pytest

def test_create_pipeline_sd15_inpaint():
    """Test creating StableDiffusion 1.5 Inpaint pipeline with ControlNet."""
    try:
        import torch
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline,
            ControlNetModel,
        )
    except ImportError:
        pytest.skip("Required dependencies not available")

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
    except ImportError:
        pytest.skip("Required dependencies not available")

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


def test_create_pipeline_qwen_image():
    """Test creating Qwen Image ControlNet pipeline."""

    import torch
    from diffusers import QwenImageControlNetModel, QwenImageMultiControlNetModel, QwenImageControlNetPipeline

    controlnet = QwenImageControlNetModel.from_pretrained(
        "InstantX/Qwen-Image-ControlNet-Union",
        torch_dtype=torch.float16,
    )
    
    pipe = QwenImageControlNetPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        controlnet=controlnet,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    assert pipe is not None

# TODO: create a flux  
