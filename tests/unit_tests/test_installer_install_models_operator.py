import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4


def _load_installer_operators_module() -> tuple[ModuleType, ModuleType]:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "installer" / "operators.py"
    module_name = f"addon_under_test.installer.operators_{uuid4().hex}"

    addon_pkg = ModuleType("addon_under_test")
    addon_pkg.__path__ = [str(repo_root)]

    installer_pkg = ModuleType("addon_under_test.installer")
    installer_pkg.__path__ = [str(repo_root / "installer")]

    bpy_module = ModuleType("bpy")
    bpy_module.types = SimpleNamespace(
        Operator=type("Operator", (), {}),
        Context=object,
    )
    bpy_module.app = SimpleNamespace(online_access=True, version=(5, 1, 0))
    bpy_module.context = SimpleNamespace(preferences=None)

    diagnostics = ModuleType("addon_under_test.diagnostics")
    diagnostics.get_log_file_path = lambda: None

    class _Logger:
        def info(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def error(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def exception(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    diagnostics.get_logger = lambda name: _Logger()

    mock_context = ModuleType("addon_under_test.mock_context")

    class MockScene:
        def __init__(self, sd_version: str = "sd15") -> None:
            self.sd_version = sd_version

    mock_context.MockScene = MockScene

    cuda = ModuleType("addon_under_test.installer.cuda")
    cuda.normalize_choice = lambda choice: choice

    paths = ModuleType("addon_under_test.installer.paths")
    paths.clean_pip_env = lambda: {}
    paths.deps_target_dir = lambda: Path("deps")
    paths.ensure_pip = lambda: None
    paths.make_importable = lambda target: None
    paths.new_deps_target_dir = lambda: Path("deps_new")
    paths.run = lambda *args, **kwargs: (0, "")
    paths.run_stream = lambda *args, **kwargs: (0, "")
    paths.set_active_deps_target = lambda target: None

    runtime_matrix = ModuleType("addon_under_test.installer.runtime_matrix")
    runtime_matrix.resolve_runtime_spec = lambda **kwargs: SimpleNamespace(
        runtime_requirements=[],
    )
    runtime_matrix.resolve_torch_install = lambda *args, **kwargs: (
        "cpu",
        "torch",
        "",
    )
    runtime_matrix.torch_index_url = lambda channel: (
        "https://download.pytorch.org/whl/cpu",
        "CPU",
    )

    model_support = importlib.import_module("model_support")

    with patch.dict(
        sys.modules,
        {
            "addon_under_test": addon_pkg,
            "addon_under_test.installer": installer_pkg,
            "addon_under_test.diagnostics": diagnostics,
            "addon_under_test.mock_context": mock_context,
            "addon_under_test.model_support": model_support,
            "addon_under_test.installer.cuda": cuda,
            "addon_under_test.installer.paths": paths,
            "addon_under_test.installer.runtime_matrix": runtime_matrix,
            "bpy": bpy_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load installer operators module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, bpy_module


def test_install_models_operator_uses_selected_model_preference() -> None:
    operators, bpy_module = _load_installer_operators_module()

    prefs = SimpleNamespace(hf_cache_path="", install_model_sd_version="sdxl")
    bpy_module.context.preferences = SimpleNamespace(
        addons={
            "addon_under_test": SimpleNamespace(preferences=prefs),
        },
    )

    diffusedtexture_pkg = ModuleType("addon_under_test.diffusedtexture")
    pipeline_pkg = ModuleType("addon_under_test.diffusedtexture.pipeline")
    pipeline_builder = ModuleType(
        "addon_under_test.diffusedtexture.pipeline.pipeline_builder",
    )
    captured: dict[str, object] = {}

    def create_diffusion_pipeline(process_parameter) -> object:  # noqa: ANN001
        captured["process_parameter"] = process_parameter
        return object()

    pipeline_builder.create_diffusion_pipeline = create_diffusion_pipeline

    reported: list[tuple[set[str], str]] = []
    operator = operators.InstallModelsOperator()
    operator.report = lambda level, message: reported.append((level, message))

    with patch.dict(
        sys.modules,
        {
            "addon_under_test.diffusedtexture": diffusedtexture_pkg,
            "addon_under_test.diffusedtexture.pipeline": pipeline_pkg,
            "addon_under_test.diffusedtexture.pipeline.pipeline_builder": pipeline_builder,
        },
    ):
        result = operator.execute(context=None)

    assert result == {"FINISHED"}
    assert captured["process_parameter"].sd_version == "sdxl"
    assert reported == [
        ({"INFO"}, "Stable Diffusion XL models installed in the default HF cache."),
    ]
