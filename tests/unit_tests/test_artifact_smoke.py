import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest


REQUIRED_PRIVATE_BUILD_EXCLUSIONS = frozenset(
    {
        ".claude/",
        ".codex/",
        ".env*",
        ".gitignore",
        ".python-version",
        ".pytest_tmp/",
        "pyproject.toml",
        "uv.lock",
    },
)


def _load_artifact_smoke() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ci"
        / "smoke_built_addon.py"
    )
    spec = importlib.util.spec_from_file_location("artifact_smoke", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load artifact smoke script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, files: dict[str, str]) -> None:
    with ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)


def _valid_files() -> dict[str, str]:
    return {
        "__init__.py": "def register(): pass\ndef unregister(): pass\n",
        "blender_manifest.toml": (
            'schema_version = "1.0.0"\n'
            'id = "diffused_texture_addon"\n'
            'version = "1.2.3"\n'
        ),
        "installer/runtime_matrix.py": "SUPPORTED = True\n",
    }


def test_manifest_excludes_private_and_development_files() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "blender_manifest.toml"
    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)

    exclusions = set(manifest["build"]["paths_exclude_pattern"])
    assert REQUIRED_PRIVATE_BUILD_EXCLUSIONS <= exclusions


def test_validate_archive_accepts_extension_root_layout(tmp_path: Path) -> None:
    smoke = _load_artifact_smoke()
    archive_path = tmp_path / "addon.zip"
    _write_archive(archive_path, _valid_files())

    members, manifest = smoke.validate_archive(archive_path)

    assert manifest["id"] == "diffused_texture_addon"
    assert manifest["version"] == "1.2.3"
    assert {member.filename for member in members} == set(_valid_files())


def test_validate_archive_rejects_wrapping_directory(tmp_path: Path) -> None:
    smoke = _load_artifact_smoke()
    archive_path = tmp_path / "wrapped.zip"
    files = {f"addon/{name}": contents for name, contents in _valid_files().items()}
    _write_archive(archive_path, files)

    with pytest.raises(ValueError, match="invalid root layout"):
        smoke.validate_archive(archive_path)


def test_validate_archive_rejects_development_paths(tmp_path: Path) -> None:
    smoke = _load_artifact_smoke()
    archive_path = tmp_path / "development-files.zip"
    files = {**_valid_files(), "tests/test_packaging.py": "pass\n"}
    _write_archive(archive_path, files)

    with pytest.raises(ValueError, match="Development-only path"):
        smoke.validate_archive(archive_path)


def test_validate_archive_rejects_parent_path(tmp_path: Path) -> None:
    smoke = _load_artifact_smoke()
    archive_path = tmp_path / "unsafe.zip"
    files = {**_valid_files(), "../outside.py": "pass\n"}
    _write_archive(archive_path, files)

    with pytest.raises(ValueError, match="Unsafe path"):
        smoke.validate_archive(archive_path)
