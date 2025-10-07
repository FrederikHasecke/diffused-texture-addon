def test_import() -> None:
    """Sanity check to importing bpy."""
    import bpy

    if bpy is None:
        msg = "Failed to import bpy module."
        raise ImportError(msg)


def test_import_diffusedtexture() -> None:
    """Sanity check to importing diffusedtexture."""
    import diffusedtexture

    if diffusedtexture is None:
        msg = "Failed to import diffusedtexture module."
        raise ImportError(msg)


def test_absolute_import() -> None:
    """Fail if any module inside the addon uses an absolute *intra-addon* import.

    Allowed:
      - Relative imports (from . / ..)
      - External modules (stdlib or third-party)
      - Blender modules: bpy, mathutils
    Disallowed:
      - Absolute imports that reference *this* addon package, e.g.:
          `from my_addon.subpkg import mod`  or  `import my_addon.mod`
    """
    import ast
    from pathlib import Path

    addon_path = Path(__file__).resolve().parent.parent
    addon_pkg_name = addon_path.name  # top-level package name == folder name

    # Folders that are not part of the shipped addon (adjust as needed)
    EXCLUDE_DIRS = {
        "tests",
        ".venv",
        "venv",
        "build",
        "dist",
        "__pycache__",
        "scratchbook",
        "scripts",
        "tools",
        "docs",
        ".mypy_cache",
        ".pytest_cache",
    }

    def is_excluded(p: Path) -> bool:
        parts = p.relative_to(addon_path).parts
        return bool(parts) and parts[0] in EXCLUDE_DIRS

    # Build set of first-party module names under the addon package
    addon_modules: set[str] = set()
    for py in addon_path.rglob("*.py"):
        if is_excluded(py):
            continue
        rel = py.relative_to(addon_path)
        parts = rel.parts
        mod = ".".join(parts).removesuffix(".py")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        dotted = (
            addon_pkg_name if mod in {"", "__init__"} else f"{addon_pkg_name}.{mod}"
        )
        addon_modules.add(dotted)

    allowed_top_level = {"bpy", "mathutils"}
    violations: list[str] = []

    def record(node, file: Path, message: str) -> None:
        lineno = getattr(node, "lineno", "?")
        violations.append(f"{file}:{lineno}: {message}")

    # Analyze each .py file via AST
    for file_path in addon_path.rglob("*.py"):
        if is_excluded(file_path):
            continue

        try:
            src = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Not relevant to import style; skip this file
            continue

        try:
            tree = ast.parse(src, filename=str(file_path))
        except SyntaxError:
            # This test only cares about import style; ignore syntax errors
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Relative import is OK
                if (getattr(node, "level", 0) or 0) > 0:
                    continue

                mod = (node.module or "").strip()
                top = mod.split(".", 1)[0] if mod else ""

                if top in allowed_top_level:
                    continue

                # Disallow absolute intra-addon imports
                if mod == addon_pkg_name or mod.startswith(addon_pkg_name + "."):
                    record(
                        node,
                        file_path,
                        f"Absolute intra-addon import: 'from {mod} import ...' "
                        f"(use relative import, e.g. 'from .subpkg import ...')",
                    )
                elif mod and any(
                    mod == am or mod.startswith(am + ".") for am in addon_modules
                ):
                    record(
                        node,
                        file_path,
                        f"Absolute intra-addon import: 'from {mod} import ...' "
                        f"(use relative import)",
                    )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    top = name.split(".", 1)[0]

                    if top in allowed_top_level:
                        continue

                    if name == addon_pkg_name or name.startswith(addon_pkg_name + "."):
                        record(
                            node,
                            file_path,
                            f"Absolute intra-addon import: 'import {name}' "
                            f"(use a relative 'from .subpkg import mod' style import)",
                        )
                    elif any(
                        name == am or name.startswith(am + ".") for am in addon_modules
                    ):
                        record(
                            node,
                            file_path,
                            f"Absolute intra-addon import: 'import {name}' "
                            f"(use relative import)",
                        )

    if violations:
        raise AssertionError(
            "Absolute intra-addon imports were found. Replace them with relative imports.\n\n"
            + "\n".join(violations)
        )
