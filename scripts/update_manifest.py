# scripts/update_manifest.py
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
wheels_dir = Path(sys.argv[2])

# Find all .whl files in wheels_dir
wheel_files = []
for whl_path in wheels_dir.rglob("*.whl"):
    rel_path = whl_path.relative_to(manifest_path.parent).as_posix()
    wheel_files.append(rel_path)

# Read manifest
content = manifest_path.read_text(encoding="utf-8")

# Replace wheels section
new_wheels = "[\n" + "\n".join(f'  "{w}",' for w in wheel_files) + "\n]"
content = re.sub(
    r"wheels\s*=\s*\[.*?\]",
    f"wheels = {new_wheels}",
    content,
    flags=re.DOTALL,
)

# Write manifest
manifest_path.write_text(content, encoding="utf-8")
