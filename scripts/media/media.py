# ruff: noqa: T201  (a CLI that reports progress on stdout)
"""Rebuild the README media from the repository.

Stages (each one can be run on its own; ``all`` runs them in order):

  check     Report Blender, the installed add-on, HDRI, fonts and ffmpeg.
  fetch     Download the CC0 Poly Haven assets listed in jobs.json (md5-checked).
  prepare   Join each scene's mesh; render it with its original albedo and as clay.
  generate  Run the add-on for each job, capture its diffusion calls, render it.
  views     Render extra stills (reverse axe) and camera-sweep frames.
  compose   Lay out the figures into images/.
  verify    Compare images/ with the checksums in documentation/*_recipe.json.

Blender stages need a GPU, the add-on installed in Blender with its Python
dependencies, and the downloaded models. ``compose`` and ``verify`` only need
Pillow and the existing workspace. See scripts/media/README.md.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMAGES = REPO / "images"
RECIPES = REPO / "documentation"
CONFIG = json.loads((HERE / "jobs.json").read_text())


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Media:
    """Pipeline stages over one media workspace."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Resolve workspace paths from the parsed command line."""
        self.work = Path(args.workdir).resolve()
        self.blender = args.blender
        self.hdri = Path(args.hdri or self.work / "assets" / CONFIG["hdri"]["file"])
        self.assets = self.work / "assets"
        self.renders = self.work / "renders"
        self.specs = self.work / "specs"
        self.logs = self.work / "logs"

    # -- helpers -----------------------------------------------------------
    def _scene(self, name: str) -> dict[str, Any]:
        return CONFIG["scenes"][name]

    def _asset_dir(self, scene: str) -> Path:
        return self.assets / self._scene(scene)["asset"]

    def _base_spec(self, name: str, scene: str) -> dict[str, Any]:
        return {
            "name": name,
            "renders_dir": str(self.renders),
            "hdri": str(self.hdri),
            "render": CONFIG["render"],
            **self._scene(scene),
        }

    def _run_blender(self, stage: str, name: str, blend: Path, spec: dict) -> None:
        if not blend.exists():
            msg = f"{stage} {name}: missing {blend}"
            raise SystemExit(msg)
        self.specs.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        spec_path = self.specs / f"{stage}-{name}.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        log = self.logs / f"{stage}-{name}.log"
        script = HERE / "blender" / f"{stage}.py"
        command = [
            self.blender, "-b", "--online-mode", str(blend),
            "--python-exit-code", "1", "--python", str(script), "--", str(spec_path),
        ]  # fmt: skip
        print(f"[{stage}] {name} -> {log}", flush=True)
        with log.open("w") as handle:
            result = subprocess.run(  # noqa: S603
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            tail = "".join(log.read_text().splitlines(keepends=True)[-15:])
            msg = f"{stage} {name} failed ({result.returncode}). Log tail:\n{tail}"
            raise SystemExit(msg)

    def _job_order(self, names: list[str]) -> list[str]:
        """Expand chained inputs so each job runs after the job it refines."""
        ordered: list[str] = []

        def visit(name: str) -> None:
            if name in ordered:
                return
            parent = CONFIG["jobs"][name].get("input_from_job")
            if parent and parent in names:
                visit(parent)
            ordered.append(name)

        for name in names:
            visit(name)
        return ordered

    # -- stages ------------------------------------------------------------
    def check(self, _: list[str]) -> None:
        """Report whether every tool and input the stages need is present."""

        def report(label: str, ok: object, detail: str) -> None:
            print(f"{'ok ' if ok else 'MISSING'}  {label:<10} {detail}")

        blender = shutil.which(self.blender)
        version = ""
        if blender:
            output = subprocess.run(  # noqa: S603
                [blender, "--version"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            version = output.splitlines()[0] if output else ""
        report("blender", blender, f"{blender} {version}")
        extensions = os.environ.get("BLENDER_USER_EXTENSIONS")
        addon = Path(extensions or "?") / "user_default" / "diffused_texture_addon"
        commit = ""
        if addon.exists():
            commit = subprocess.run(  # noqa: S603
                ["git", "-C", str(addon), "rev-parse", "--short", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        report("add-on", addon.exists(), f"{addon} {commit or '(not a git checkout)'}")
        hdri_ok = self.hdri.exists() and digest(self.hdri) == CONFIG["hdri"]["sha256"]
        report("hdri", hdri_ok, f"{self.hdri} (sha256 must match jobs.json)")
        from style import FONTS

        missing = [f for f in FONTS if not Path(f).exists()]
        report("fonts", not missing, ", ".join(missing) or "Ubuntu R/M/B")
        report("ffmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "")
        report("workdir", self.work.exists(), str(self.work))

    def fetch(self, names: list[str]) -> None:
        """Download missing or changed Poly Haven asset files."""
        for asset in names or CONFIG["assets"]:
            for rel, source in CONFIG["assets"][asset]["files"].items():
                target = self.assets / asset / rel
                if target.exists() and digest(target, "md5") == source["md5"]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"[fetch] {source['url']}", flush=True)
                partial = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(source["url"], partial)  # noqa: S310
                if digest(partial, "md5") != source["md5"]:
                    partial.unlink()
                    msg = f"md5 mismatch for {source['url']}"
                    raise SystemExit(msg)
                partial.replace(target)

    def prepare(self, names: list[str]) -> None:
        """Prepare scenes and render their original-albedo and clay versions."""
        for scene in names or CONFIG["scenes"]:
            asset = self._scene(scene)["asset"]
            folder = self._asset_dir(scene)
            spec = self._base_spec(scene, scene)
            spec["prepared_blend"] = str(self.assets / f"{scene}_prepared.blend")
            spec["albedo"] = str(next((folder / "textures").glob("*_diff_2k.jpg")))
            self._run_blender("prepare", scene, folder / f"{asset}.blend", spec)

    def generate(self, names: list[str]) -> None:
        """Generate textures with the add-on, in chained-input order."""
        for name in self._job_order(names or list(CONFIG["jobs"])):
            job = CONFIG["jobs"][name]
            spec = self._base_spec(name, job["scene"])
            spec |= CONFIG["defaults"] | job
            spec["addon_module"] = CONFIG["addon_module"]
            texture = job.get("input_texture", "")
            if texture.startswith("asset:"):
                texture = str(
                    self._asset_dir(job["scene"]) / texture.removeprefix("asset:"),
                )
            if "input_from_job" in job:
                from figures import Workspace

                texture = str(Workspace(self.work).texture(job["input_from_job"]))
            spec["input_texture"] = texture or None
            if spec.pop("capture_pipeline"):
                capture = self.work / "pipeline_capture" / name
                shutil.rmtree(capture, ignore_errors=True)
                spec["capture_dir"] = str(capture)
            blend = self.assets / f"{job['scene']}_prepared.blend"
            self._run_blender("generate", name, blend, spec)

    def views(self, names: list[str]) -> None:
        """Render extra stills and camera-sweep frames of generated scenes."""
        for name in names or list(CONFIG["jobs"]):
            job = CONFIG["jobs"][name]
            if not job.get("views") and not job.get("sweep"):
                continue
            spec = self._base_spec(name, job["scene"])
            spec |= {
                "object": job["scene"],
                "views": job.get("views", {}),
                "sweep": CONFIG["sweep"] if job.get("sweep") else None,
                "frames_dir": str(self.work / "frames" / job["scene"]),
            }
            self._run_blender("views", name, self.renders / f"{name}.blend", spec)

    def compose(self, names: list[str]) -> None:
        """Lay out the figures whose workspace inputs exist."""
        from figures import FIGURES, Workspace

        work = Workspace(self.work)
        for name in names or FIGURES:
            layout, needs = FIGURES[name]
            missing = [n for n in needs if not (self.work / n).exists()]
            if missing:
                print(f"[compose] skip {name}: missing {', '.join(missing)}")
                continue
            for path in layout(work, IMAGES):
                print(f"[compose] {path.relative_to(REPO)}")

    def verify(self, _: list[str]) -> None:
        """Compare images/ with the checksums recorded in the recipes."""
        recorded = recorded_checksums()
        status = 0
        for path in sorted(IMAGES.iterdir()):
            expected = recorded.get(path.name)
            if expected is None:
                continue
            same = digest(path) == expected
            status |= not same
            print(f"{'same   ' if same else 'CHANGED'}  images/{path.name}")
        unrecorded = sorted(
            f"images/{p.name}"
            for p in IMAGES.iterdir()
            if p.is_file() and p.name not in recorded
        )
        print("not recorded in a recipe:", ", ".join(unrecorded) or "none")
        if status:
            print("Update the recipe checksums if the change is intentional.")
        sys.exit(status)


def recorded_checksums() -> dict[str, str]:
    """Collect the image checksums recorded in documentation/*_recipe.json."""
    found: dict[str, str] = {}
    for recipe in sorted(RECIPES.glob("*_recipe.json")):
        data = json.loads(recipe.read_text())
        for name, entry in data.get("outputs", {}).items():
            found[name] = entry["sha256"]
        image = data.get("comparison_image")
        if image:
            found[Path(image["path"]).name] = image["sha256"]
    return found


STAGES = ("fetch", "prepare", "generate", "views", "compose")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=("check", *STAGES, "verify", "all"))
    parser.add_argument(
        "names",
        nargs="*",
        help="assets / scenes / jobs / figures to limit the stage to (default: all)",
    )
    parser.add_argument(
        "--workdir",
        default=os.environ.get("DT_MEDIA_WORKDIR"),
        help="media workspace (default: $DT_MEDIA_WORKDIR)",
    )
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument(
        "--hdri",
        help="studio HDRI (default: <workdir>/assets/studio.hdr)",
    )
    args = parser.parse_args()
    if not args.workdir:
        parser.error("--workdir or DT_MEDIA_WORKDIR is required")
    sys.path.insert(0, str(HERE))
    media = Media(args)
    if args.stage == "all":
        if args.names:
            parser.error("'all' always runs every asset, scene, job and figure")
        for stage in STAGES:
            getattr(media, stage)([])
    else:
        getattr(media, args.stage)(args.names)


if __name__ == "__main__":
    main()
