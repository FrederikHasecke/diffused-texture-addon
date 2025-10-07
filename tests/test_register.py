def test_register_unregister_bare() -> None:
    # Import the register module
    import diffused_texture_addon

    # Call register
    diffused_texture_addon.register()

    # Call unregister
    diffused_texture_addon.unregister()
