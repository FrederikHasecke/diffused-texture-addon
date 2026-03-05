import pytest


@pytest.mark.slow
def test_dependency_imports() -> None:
    try:
        import accelerate
        import cv2
        import diffusers
        import numpy as np
        import peft
        import PIL
        import safetensors
        import torch
        import transformers
    except ImportError as e:
        pytest.fail(str(e))
