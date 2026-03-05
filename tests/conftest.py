import os
import sys
import pytest

# Add the parent directory of the addon package to Python path
# this ensures `import diffused_texture_addon` works; previously we
# were adding the package directory itself which meant Python could not
# locate a subdirectory of that name, causing ModuleNotFoundError.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_parent = os.path.dirname(root_dir)
sys.path.insert(0, package_parent)


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
