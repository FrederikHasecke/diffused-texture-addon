import importlib
import pytest


def test_register_unregister_bare() -> None:
    """Test registering and unregistering the addon module."""
    try:
        import bpy
    except ImportError:
        pytest.fail("bpy module not available")
    
    # Clean file for deterministic tests
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Enable addon from venv (importable module), no zip/install step
    mod = importlib.import_module("diffused_texture_addon")
    if hasattr(mod, "register"):
        mod.register()
    else:
        raise ImportError("Module does not have a register function")

    # Optional: clean up
    if hasattr(mod, "unregister"):
        mod.unregister()
    else:
        raise ImportError("Module does not have an unregister function")

    
def test_register_unregister_bare_2() -> None:
    """Test registering and unregistering using direct import."""
    try:
        import bpy
    except ImportError:
        pytest.fail("bpy module not available")
    
    # Import the register module
    import diffused_texture_addon

    # Call register
    diffused_texture_addon.register()

    # Call unregister
    diffused_texture_addon.unregister()
