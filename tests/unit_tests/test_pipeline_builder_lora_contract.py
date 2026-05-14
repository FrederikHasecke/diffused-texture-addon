import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4


class _FakePipeline:
    def __init__(self) -> None:
        self.load_lora_weights_calls: list[tuple[tuple, dict]] = []
        self.fuse_lora_calls: list[dict] = []
        self.to_calls: list[str] = []
        self.cpu_offload_calls = 0

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN002, ANN003
        return cls()

    @classmethod
    def from_single_file(cls, *args, **kwargs):  # noqa: ANN002, ANN003
        return cls()

    def load_lora_weights(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.load_lora_weights_calls.append((args, kwargs))

    def fuse_lora(self, **kwargs):  # noqa: ANN003
        self.fuse_lora_calls.append(kwargs)

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def enable_model_cpu_offload(self) -> None:
        self.cpu_offload_calls += 1


def _fake_torch_module(
    *,
    cuda_available: bool = False,
    mps_available: bool = False,
) -> ModuleType:
    torch = ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return mps_available

    class _Backends:
        mps = _Mps()

    torch.cuda = _Cuda()
    torch.backends = _Backends()
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    return torch


def _load_pipeline_builder() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diffusedtexture" / "pipeline" / "pipeline_builder.py"
    module_name = f"diffused_texture_addon.diffusedtexture.pipeline.pipeline_builder_{uuid4().hex}"

    addon_pkg = ModuleType("diffused_texture_addon")
    addon_pkg.__path__ = [str(repo_root)]

    diffusedtexture_pkg = ModuleType("diffused_texture_addon.diffusedtexture")
    diffusedtexture_pkg.__path__ = [str(repo_root / "diffusedtexture")]

    pipeline_pkg = ModuleType("diffused_texture_addon.diffusedtexture.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "diffusedtexture" / "pipeline")]

    blender_ops = ModuleType("diffused_texture_addon.blender_operations")
    blender_ops.ProcessParameter = object

    controlnet_config = ModuleType(
        "diffused_texture_addon.diffusedtexture.pipeline.controlnet_config",
    )
    controlnet_config.load_controlnet_models = lambda process_parameter: [
        "fake_controlnet",
    ]
    runtime_capability = ModuleType("diffused_texture_addon.runtime_capability")

    def _resolve_diffusion_device(torch_module=None):  # noqa: ANN001
        if torch_module is None:
            return None
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return "cuda"
        if (
            getattr(torch_module, "backends", None)
            and getattr(torch_module.backends, "mps", None)
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    runtime_capability.resolve_diffusion_device = _resolve_diffusion_device

    with patch.dict(
        sys.modules,
        {
            "diffused_texture_addon": addon_pkg,
            "diffused_texture_addon.diffusedtexture": diffusedtexture_pkg,
            "diffused_texture_addon.diffusedtexture.pipeline": pipeline_pkg,
            "diffused_texture_addon.blender_operations": blender_ops,
            "diffused_texture_addon.diffusedtexture.pipeline.controlnet_config": controlnet_config,
            "diffused_texture_addon.runtime_capability": runtime_capability,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load pipeline_builder module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_create_pipeline_uses_lora_dict_for_loading_arguments() -> None:
    pipeline_builder = _load_pipeline_builder()

    diffusers = ModuleType("diffusers")
    diffusers.StableDiffusionControlNetInpaintPipeline = _FakePipeline
    diffusers.StableDiffusionXLControlNetUnionInpaintPipeline = _FakePipeline

    process_parameter = SimpleNamespace(
        sd_version="sd15",
        dtype="float16",
        mesh_complexity="HIGH",
        checkpoint_path="runwayml/stable-diffusion-v1-5",
        num_loras=1,
        lora_models=[{"path": "loras/style.safetensors", "strength": 0.7}],
        use_ipadapter=False,
        ipadapter_strength=0.0,
    )

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch_module(),
            "diffusers": diffusers,
        },
    ):
        pipe = pipeline_builder.create_diffusion_pipeline(process_parameter)

    assert pipe.load_lora_weights_calls == [
        (
            (str(Path("loras/style.safetensors").parent),),
            {"weight_name": "style.safetensors"},
        ),
    ]
    assert pipe.fuse_lora_calls == [{"lora_scale": 0.7}]


def test_create_pipeline_applies_each_lora_with_matching_scale() -> None:
    pipeline_builder = _load_pipeline_builder()

    diffusers = ModuleType("diffusers")
    diffusers.StableDiffusionControlNetInpaintPipeline = _FakePipeline
    diffusers.StableDiffusionXLControlNetUnionInpaintPipeline = _FakePipeline

    lora_paths = [
        "assets/loras/character.safetensors",
        "assets/loras/lighting.safetensors",
    ]
    process_parameter = SimpleNamespace(
        sd_version="sd15",
        dtype="float16",
        mesh_complexity="HIGH",
        checkpoint_path="runwayml/stable-diffusion-v1-5",
        num_loras=2,
        lora_models=[
            {"path": lora_paths[0], "strength": 0.45},
            {"path": lora_paths[1], "strength": 0.9},
        ],
        use_ipadapter=False,
        ipadapter_strength=0.0,
    )

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch_module(),
            "diffusers": diffusers,
        },
    ):
        pipe = pipeline_builder.create_diffusion_pipeline(process_parameter)

    assert pipe.load_lora_weights_calls == [
        (
            (str(Path(lora_paths[0]).parent),),
            {"weight_name": Path(lora_paths[0]).name},
        ),
        (
            (str(Path(lora_paths[1]).parent),),
            {"weight_name": Path(lora_paths[1]).name},
        ),
    ]
    assert pipe.fuse_lora_calls == [
        {"lora_scale": 0.45},
        {"lora_scale": 0.9},
    ]


def test_create_pipeline_uses_cpu_without_offload_when_no_accelerator() -> None:
    pipeline_builder = _load_pipeline_builder()

    diffusers = ModuleType("diffusers")
    diffusers.StableDiffusionControlNetInpaintPipeline = _FakePipeline
    diffusers.StableDiffusionXLControlNetUnionInpaintPipeline = _FakePipeline

    process_parameter = SimpleNamespace(
        sd_version="sd15",
        dtype="float16",
        mesh_complexity="HIGH",
        checkpoint_path="runwayml/stable-diffusion-v1-5",
        num_loras=0,
        lora_models=[],
        use_ipadapter=False,
        ipadapter_strength=0.0,
    )

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch_module(),
            "diffusers": diffusers,
        },
    ):
        pipe = pipeline_builder.create_diffusion_pipeline(process_parameter)

    assert pipe.to_calls == ["cpu"]
    assert pipe.cpu_offload_calls == 0


def test_create_pipeline_uses_cuda_and_cpu_offload_when_available() -> None:
    pipeline_builder = _load_pipeline_builder()

    diffusers = ModuleType("diffusers")
    diffusers.StableDiffusionControlNetInpaintPipeline = _FakePipeline
    diffusers.StableDiffusionXLControlNetUnionInpaintPipeline = _FakePipeline

    process_parameter = SimpleNamespace(
        sd_version="sd15",
        dtype="float16",
        mesh_complexity="HIGH",
        checkpoint_path="runwayml/stable-diffusion-v1-5",
        num_loras=0,
        lora_models=[],
        use_ipadapter=False,
        ipadapter_strength=0.0,
    )

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch_module(cuda_available=True),
            "diffusers": diffusers,
        },
    ):
        pipe = pipeline_builder.create_diffusion_pipeline(process_parameter)

    assert pipe.to_calls == ["cuda"]
    assert pipe.cpu_offload_calls == 1


def test_create_pipeline_uses_mps_and_cpu_offload_when_cuda_unavailable() -> None:
    pipeline_builder = _load_pipeline_builder()

    diffusers = ModuleType("diffusers")
    diffusers.StableDiffusionControlNetInpaintPipeline = _FakePipeline
    diffusers.StableDiffusionXLControlNetUnionInpaintPipeline = _FakePipeline

    process_parameter = SimpleNamespace(
        sd_version="sd15",
        dtype="float16",
        mesh_complexity="HIGH",
        checkpoint_path="runwayml/stable-diffusion-v1-5",
        num_loras=0,
        lora_models=[],
        use_ipadapter=False,
        ipadapter_strength=0.0,
    )

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch_module(mps_available=True),
            "diffusers": diffusers,
        },
    ):
        pipe = pipeline_builder.create_diffusion_pipeline(process_parameter)

    assert pipe.to_calls == ["mps"]
    assert pipe.cpu_offload_calls == 1
