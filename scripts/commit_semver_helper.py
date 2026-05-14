"""Shared Git hook helpers for Conventional Commits and semver guidance."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VALID_TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)

VALID_TYPES_PATTERN = "|".join(VALID_TYPES)
SUBJECT_PATTERN = re.compile(
    rf"^(?P<type>{VALID_TYPES_PATTERN})"
    rf"(\([a-z0-9][a-z0-9._/-]*\))?"
    rf"(?P<breaking>!)?: (?P<subject>.+)$",
)
BREAKING_PATTERN = re.compile(r"^BREAKING CHANGE: ", re.MULTILINE)

COMMENT_BLOCK = (
    "# Conventional Commit quick guide\n"
    "# feat: add a new backward-compatible capability -> minor release\n"
    "# fix: correct broken behavior -> patch release\n"
    "# feat!: or fix!: incompatible change -> major release\n"
    "# docs:, test:, chore:, refactor:, ci:, build:, style: -> no release by default\n"
    "# Examples:\n"
    "#   feat(ui): add SDXL preset selector\n"
    "#   fix(installer): resolve Blender 5.1 pip wheel selection\n"
    "#   feat!: drop Blender 4.x support\n"
    "#   docs(releasing): document release bootstrap\n\n"
)


def _read_message(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_message(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _first_non_comment_line(message: str) -> str:
    for line in message.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return ""


def _breaking_change(message: str) -> bool:
    return BREAKING_PATTERN.search(message) is not None


def _release_impact(subject: str, message: str) -> str:
    if subject.startswith("Merge "):
        return "merge"
    match = SUBJECT_PATTERN.match(subject)
    if not match:
        return "invalid"
    if match.group("breaking") or _breaking_change(message):
        return "major"
    commit_type = match.group("type")
    if commit_type == "feat":
        return "minor"
    if commit_type in {"fix", "perf"}:
        return "patch"
    return "none"


def _prompt(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/N]: ").strip().lower()
        if answer in {"", "n", "no"}:
            return False
        if answer in {"y", "yes"}:
            return True

        _write_stdout("Please answer with 'y' or 'n'.\n")


def _write_stdout(message: str) -> None:
    sys.stdout.write(message)


def _write_stderr(message: str) -> None:
    sys.stderr.write(message)


def run_pre_commit(_: argparse.Namespace) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return 0

    _write_stdout("SemVer commit helper\n")
    _write_stdout(
        "Use this short checklist before the commit message is finalized.\n\n",
    )

    breaking = _prompt(
        "Does this change break existing behavior or drop supported "
        "Blender/Python/runtime compatibility?",
    )
    feature = _prompt(
        "Does this add a new user-visible capability, model support, or workflow?",
    )
    fix = _prompt(
        "Does this fix broken runtime, packaging, installer, or user-facing behavior?",
    )
    internal_only = _prompt(
        "Is this mainly docs, tests, CI, refactoring, or chore work without "
        "behavior changes?",
    )

    _write_stdout("\n")
    if breaking:
        _write_stdout("Recommended release impact: major\n")
        _write_stdout("Use a Conventional Commit with '!'.\n")
        if feature:
            _write_stdout("Recommended prefix: feat!:\n")
        elif fix:
            _write_stdout("Recommended prefix: fix!:\n")
        else:
            _write_stdout("Recommended prefix: refactor!: or chore!:\n")
        _write_stdout(
            "Add a 'BREAKING CHANGE:' footer when the migration detail matters.\n",
        )
    elif feature:
        _write_stdout("Recommended release impact: minor\n")
        _write_stdout("Recommended prefix: feat:\n")
    elif fix:
        _write_stdout("Recommended release impact: patch\n")
        _write_stdout("Recommended prefix: fix:\n")
    elif internal_only:
        _write_stdout("Recommended release impact: none by default\n")
        _write_stdout(
            "Recommended prefixes: docs:, test:, chore:, refactor:, ci:, build:, "
            "or style:\n",
        )
    else:
        _write_stdout("No clear semver impact was selected.\n")
        _write_stdout(
            "Pick the commit type that best matches the primary change and keep "
            "the subject specific.\n",
        )

    _write_stdout("\n")
    _write_stdout(
        "The commit-msg hook will validate the final Conventional Commit format.\n",
    )
    input("Press Enter to continue with the commit, or Ctrl+C to cancel. ")
    return 0


def run_prepare_commit_msg(args: argparse.Namespace) -> int:
    if args.source in {"message", "merge", "squash", "commit"}:
        return 0

    message_path = Path(args.message_file)
    current = _read_message(message_path)
    if "# Conventional Commit quick guide" in current:
        return 0

    _write_message(message_path, COMMENT_BLOCK + current)
    return 0


def run_commit_msg(args: argparse.Namespace) -> int:
    message_path = Path(args.message_file)
    message = _read_message(message_path)
    subject = _first_non_comment_line(message)

    if not subject:
        _write_stderr("Commit message is empty.\n")
        return 1

    if subject.startswith("Merge "):
        _write_stderr(
            "Merge commit message detected. It is allowed locally, but "
            "release-please only bumps versions from Conventional Commits.\n",
        )
        return 0

    if subject.startswith("Revert "):
        _write_stderr(
            "Auto-generated revert commit detected. Consider rewriting it to "
            "'revert:' if it should drive release notes.\n",
        )
        return 0

    match = SUBJECT_PATTERN.match(subject)
    if not match:
        _write_stderr("Invalid commit message format.\n")
        _write_stderr(
            "Expected: type(scope optional): subject or type(scope optional)!: "
            "subject\n",
        )
        _write_stderr(
            "Valid types: build, chore, ci, docs, feat, fix, perf, refactor, "
            "revert, style, test\n",
        )
        _write_stderr("Examples:\n")
        _write_stderr("  feat(ui): add SDXL preset selector\n")
        _write_stderr(
            "  fix(installer): resolve Blender 5.1 pip wheel selection\n",
        )
        _write_stderr("  feat!: drop Blender 4.x support\n")
        return 1

    impact = _release_impact(subject, message)
    if impact == "major":
        _write_stdout("Commit accepted: release impact major\n")
    elif impact == "minor":
        _write_stdout("Commit accepted: release impact minor\n")
    elif impact == "patch":
        _write_stdout("Commit accepted: release impact patch\n")
    else:
        _write_stdout("Commit accepted: no release bump by default\n")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("pre-commit").set_defaults(func=run_pre_commit)

    prepare = subparsers.add_parser("prepare-commit-msg")
    prepare.add_argument("message_file")
    prepare.add_argument("source", nargs="?", default="")
    prepare.add_argument("sha", nargs="?", default="")
    prepare.set_defaults(func=run_prepare_commit_msg)

    commit_msg = subparsers.add_parser("commit-msg")
    commit_msg.add_argument("message_file")
    commit_msg.set_defaults(func=run_commit_msg)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
