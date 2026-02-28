import os
import sys
import pytest

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)


def pytest_configure(config):
    """Pytest hook to configure test discovery."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests during normal test discovery to prevent timeouts."""
    # Check if user explicitly requested slow tests with pytest -m slow
    if config.getoption("-m") and "slow" in config.getoption("-m"):
        return
    
    # Otherwise, skip slow tests by default
    skip_slow = pytest.mark.skip(reason="Slow test - run with pytest -m slow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)