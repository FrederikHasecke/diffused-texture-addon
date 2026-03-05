from .model_support import DEFAULT_SD_VERSION, get_default_model_paths


class MockScene:
    """Mockup Scene."""

    def __init__(self) -> None:
        """Initialize a Mock-up Scene."""
        self.num_loras = 0
        self.use_ipadapter = True
        self.ipadapter_strength = 0.5
        self.mesh_complexity = "HIGH"
        self.depth_controlnet_strength = 1.0
        self.canny_controlnet_strength = 1.0
        self.normal_controlnet_strength = 1.0
        self.sd_version = DEFAULT_SD_VERSION
        defaults = get_default_model_paths(self.sd_version)
        self.checkpoint_path = defaults["checkpoint_path"]
        self.dtype = "float16"
        self.controlnet_union_path = defaults["controlnet_union_path"]
        self.canny_controlnet_path = defaults["canny_controlnet_path"]
        self.normal_controlnet_path = defaults["normal_controlnet_path"]
        self.depth_controlnet_path = defaults["depth_controlnet_path"]


class MockUpContext:
    """Mockup Context used for default model download."""

    def __init__(self) -> None:
        """Initialize a Mock-up Context."""
        self.scene = MockScene()
