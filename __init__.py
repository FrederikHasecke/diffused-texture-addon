"""====================== BEGIN GPL LICENSE BLOCK ======================.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

======================= END GPL LICENSE BLOCK ========================
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

import bpy

_MINIMAL_PREFS_ONLY = False  # set True if we fall back to registering only Preferences


def _deps_target_dir() -> Path:
    base = Path(bpy.utils.user_resource("SCRIPTS", path="", create=True))
    return base / "modules" / "diffusedtexture_deps"


# Treat the deps dir as a site dir so .pth inside it (if any) are processed.
deps = _deps_target_dir()
if deps.exists():
    site.addsitedir(str(deps))  # adds to sys.path (and processes any .pth)
else:
    # still add so later install in this path is immediately importable in-session
    sys.path.insert(0, str(deps))


def register():
    """Register the add-on inside Blender."""
    global _MINIMAL_PREFS_ONLY
    try:
        from .registration import register_addon

        register_addon()
        _MINIMAL_PREFS_ONLY = False
        print("Full addon registration successful")
    except Exception as e:
        print(f"Full registration failed, falling back to minimal mode: {str(e)}")
        from .preferences import register as register_prefs

        register_prefs()
        _MINIMAL_PREFS_ONLY = True


def unregister() -> None:
    """Unregister the add-on."""
    if _MINIMAL_PREFS_ONLY:
        from .preferences import unregister as unregister_prefs

        unregister_prefs()
    else:
        from .registration import unregister_addon

        unregister_addon()


if __name__ == "__main__":
    register()
