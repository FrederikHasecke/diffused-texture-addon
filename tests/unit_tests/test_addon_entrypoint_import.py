import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_addon_entrypoint_imports_without_package_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "__init__.py"

    bpy_module = ModuleType("bpy")
    bpy_module.utils = SimpleNamespace(
        user_resource=lambda *_args, **_kwargs: str(tmp_path / "scripts"),
    )

    preserved_modules = {
        name: sys.modules.get(name)
        for name in ("__init__", "bpy", "diagnostics", "installer", "installer.paths")
    }

    monkeypatch.syspath_prepend(str(repo_root))
    for name in preserved_modules:
        sys.modules.pop(name, None)
    sys.modules["bpy"] = bpy_module

    try:
        spec = importlib.util.spec_from_file_location("__init__", module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load addon entrypoint module spec."
            raise RuntimeError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.activate_deps_import_path is not None
        assert module.deps_target_dir() == tmp_path / "scripts" / "modules" / "diffusedtexture_deps"
    finally:
        for name in preserved_modules:
            sys.modules.pop(name, None)
        for name, preserved in preserved_modules.items():
            if preserved is not None:
                sys.modules[name] = preserved
