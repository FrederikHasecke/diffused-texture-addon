import importlib.abc
import importlib.util
import sys
from pathlib import Path

import pytest


class _AddonModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir
        self._module_name = "diffused_texture_addon"

    def find_spec(self, fullname: str, path, target=None):
        del path, target

        if fullname == self._module_name:
            return importlib.util.spec_from_file_location(
                fullname,
                self._root_dir / "__init__.py",
                submodule_search_locations=[str(self._root_dir)],
            )

        prefix = f"{self._module_name}."
        if not fullname.startswith(prefix):
            return None

        relative_parts = fullname[len(prefix) :].split(".")
        module_path = self._root_dir.joinpath(*relative_parts)
        package_init = module_path / "__init__.py"
        if package_init.is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                package_init,
                submodule_search_locations=[str(module_path)],
            )

        module_file = module_path.with_suffix(".py")
        if module_file.is_file():
            return importlib.util.spec_from_file_location(fullname, module_file)

        return None


def _install_addon_module_finder() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    finder = _AddonModuleFinder(root_dir)
    sys.meta_path.insert(0, finder)


_install_addon_module_finder()


def pytest_configure(config):
    """Pytest hook to configure test discovery."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "exclusive: marks tests that must run serially with other exclusive tests",
    )


def pytest_collection_modifyitems(config, items):
    """Group exclusive tests onto a single xdist worker."""
    del config
    for item in items:
        if item.get_closest_marker("exclusive") is not None:
            item.add_marker(pytest.mark.xdist_group("exclusive"))
