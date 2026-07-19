"""Validate and register the packaged add-on without diffusion dependencies."""

from __future__ import annotations

import argparse
import importlib
import py_compile
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile, ZipInfo

if TYPE_CHECKING:
    from types import ModuleType

EXPECTED_ADDON_ID = "diffused_texture_addon"
REQUIRED_ROOT_FILES = frozenset({"__init__.py", "blender_manifest.toml"})
FORBIDDEN_ROOTS = frozenset(
    {
        ".git",
        ".github",
        ".gitignore",
        ".env",
        ".python-version",
        ".pytest_tmp",
        ".venv",
        ".claude",
        "Dockerfile",
        "documentation",
        "images",
        "pyproject.toml",
        "pytest.ini",
        "scratchbook",
        "scripts",
        "tests",
        "uv.lock",
    },
)


def _member_path(info: ZipInfo) -> PurePosixPath:
    """Return a safe normalized archive member path."""
    normalized = info.filename.replace("\\", "/")
    if info.is_dir() and normalized.rstrip("/") in {"", "."}:
        return PurePosixPath(".")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        msg = f"Unsafe path in add-on archive: {info.filename}"
        raise ValueError(msg)
    return path


def validate_archive(archive_path: Path) -> tuple[list[ZipInfo], dict[str, Any]]:
    """Validate the distributable layout and return safe file members."""
    if not archive_path.is_file():
        msg = f"Add-on archive not found: {archive_path}"
        raise FileNotFoundError(msg)

    with ZipFile(archive_path) as archive:
        members: list[ZipInfo] = []
        names: dict[str, ZipInfo] = {}
        for info in archive.infolist():
            path = _member_path(info)
            normalized = path.as_posix()
            if normalized in names:
                msg = f"Duplicate path in add-on archive: {normalized}"
                raise ValueError(msg)
            names[normalized] = info
            if info.is_dir():
                continue
            if path.parts[0] in FORBIDDEN_ROOTS:
                msg = f"Development-only path included in add-on archive: {normalized}"
                raise ValueError(msg)
            members.append(info)

        missing = REQUIRED_ROOT_FILES.difference(names)
        if missing:
            missing_text = ", ".join(sorted(missing))
            msg = (
                "Add-on archive has an invalid root layout; missing root files: "
                f"{missing_text}"
            )
            raise ValueError(msg)

        manifest = tomllib.loads(
            archive.read(names["blender_manifest.toml"]).decode("utf-8"),
        )

    addon_id = manifest.get("id")
    if addon_id != EXPECTED_ADDON_ID:
        msg = f"Unexpected add-on id {addon_id!r}; expected {EXPECTED_ADDON_ID!r}"
        raise ValueError(msg)

    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        msg = "blender_manifest.toml must contain a non-empty version"
        raise ValueError(msg)

    return members, manifest


def extract_archive(
    archive_path: Path,
    members: list[ZipInfo],
    destination: Path,
) -> None:
    """Extract previously validated members into an isolated package directory."""
    with ZipFile(archive_path) as archive:
        for info in members:
            relative_path = _member_path(info)
            target = destination.joinpath(*relative_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def compile_packaged_python(package_dir: Path) -> None:
    """Compile every packaged Python source file."""
    python_files = sorted(package_dir.rglob("*.py"))
    if not python_files:
        msg = "Add-on archive does not contain Python source files"
        raise ValueError(msg)
    for python_file in python_files:
        py_compile.compile(str(python_file), doraise=True)


def _without_repo_root(repo_root: Path) -> list[str]:
    """Return sys.path entries that cannot resolve imports from the checkout."""
    cleaned: list[str] = []
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if candidate == repo_root:
            continue
        cleaned.append(entry)
    return cleaned


def register_packaged_addon(package_parent: Path) -> Path:
    """Import, fully register, and unregister the isolated packaged add-on."""
    import bpy

    repo_root = Path(__file__).resolve().parents[2]
    original_path = sys.path.copy()
    registered = False
    module: ModuleType | None = None
    try:
        sys.path[:] = [str(package_parent), *_without_repo_root(repo_root)]
        importlib.invalidate_caches()
        module = importlib.import_module(EXPECTED_ADDON_ID)
        module_path = Path(str(module.__file__)).resolve()
        expected_package_dir = (package_parent / EXPECTED_ADDON_ID).resolve()
        if not module_path.is_relative_to(expected_package_dir):
            msg = f"Smoke test imported checkout instead of artifact: {module_path}"
            raise RuntimeError(msg)

        bpy.ops.wm.read_factory_settings(use_empty=True)
        module.register()
        registered = True
        if bool(getattr(module, "_MINIMAL_PREFS_ONLY", True)):
            msg = "Packaged add-on fell back to minimal preferences registration"
            raise RuntimeError(msg)
        if not hasattr(bpy.types.Scene, "sd_version"):
            msg = "Full add-on registration did not create Scene.sd_version"
            raise RuntimeError(msg)
        return module_path
    finally:
        if registered and module is not None:
            module.unregister()
            if hasattr(bpy.types.Scene, "sd_version"):
                msg = "Add-on unregistration left Scene.sd_version registered"
                raise RuntimeError(msg)
        sys.path[:] = original_path


def smoke_archive(archive_path: Path) -> tuple[str, Path]:
    """Run all lightweight artifact validation and registration checks."""
    members, manifest = validate_archive(archive_path)
    version = str(manifest["version"])
    with tempfile.TemporaryDirectory(prefix="diffused_texture_artifact_") as temp:
        package_parent = Path(temp) / "installed"
        package_dir = package_parent / EXPECTED_ADDON_ID
        package_dir.mkdir(parents=True)
        extract_archive(archive_path, members, package_dir)
        compile_packaged_python(package_dir)
        module_path = register_packaged_addon(package_parent)
    return version, module_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and register a built DiffusedTexture add-on ZIP.",
    )
    parser.add_argument("archive", type=Path, help="Path to the built add-on ZIP")
    return parser.parse_args()


def main() -> int:
    """Run the command-line artifact smoke test."""
    args = _parse_args()
    version, module_path = smoke_archive(args.archive.resolve())
    sys.stdout.write(
        f"Artifact smoke passed: version={version} module={module_path}\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
