import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4


def _load_paths_module(tmp_path: Path) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "installer" / "paths.py"
    module_name = f"installer_paths_under_test_{uuid4().hex}"

    bpy_module = ModuleType("bpy")
    bpy_module.utils = SimpleNamespace(
        user_resource=lambda *args, **kwargs: str(tmp_path),  # noqa: ARG005
    )

    diagnostics_module = ModuleType("diagnostics")

    class _Logger:
        def debug(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return

        def exception(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return

    diagnostics_module.get_logger = lambda name: _Logger()  # noqa: ARG005

    with patch.dict(
        sys.modules,
        {
            "bpy": bpy_module,
            "diagnostics": diagnostics_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load installer.paths module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def test_activate_deps_import_path_promotes_env_and_dedupes(tmp_path: Path) -> None:
    paths = _load_paths_module(tmp_path)
    env_dir = tmp_path / "env_active"
    env_dir.mkdir(parents=True, exist_ok=True)
    other_a = tmp_path / "other_a"
    other_b = tmp_path / "other_b"
    other_a.mkdir()
    other_b.mkdir()

    original_path = list(sys.path)
    try:
        sys.path[:] = [
            str(other_a),
            str(env_dir),
            str(other_b),
            str(env_dir),
            str(other_a),
        ]

        paths.activate_deps_import_path(env_dir)

        assert sys.path[0] == str(env_dir)
        assert sum(1 for value in sys.path if value == str(env_dir)) == 1
        assert sum(1 for value in sys.path if value == str(other_a)) == 1
    finally:
        sys.path[:] = original_path


def test_activate_deps_import_path_handles_missing_target(tmp_path: Path) -> None:
    paths = _load_paths_module(tmp_path)
    missing_dir = tmp_path / "env_missing"

    original_path = list(sys.path)
    try:
        sys.path[:] = [str(tmp_path / "x")]

        paths.activate_deps_import_path(missing_dir)

        assert sys.path[0] == str(missing_dir)
    finally:
        sys.path[:] = original_path
