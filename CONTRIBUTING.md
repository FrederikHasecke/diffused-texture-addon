# Contributing

## Commit Setup

This repository ships repo-tracked Git hooks under `.githooks/`.

Run one of these once per clone:

1. PowerShell: `./scripts/setup-git-hooks.ps1`
2. POSIX shell: `sh ./scripts/setup-git-hooks.sh`

After setup:

1. `pre-commit` asks the semver questions before every commit.
2. `prepare-commit-msg` injects a short Conventional Commit reminder into the editor.
3. `commit-msg` rejects invalid commit titles.

## Conventional Commits

Use Conventional Commits for every commit that should influence a release:

1. `feat:` adds a backward-compatible capability and triggers a minor release.
2. `fix:` corrects broken behavior and triggers a patch release.
3. `type!:` or a `BREAKING CHANGE:` footer marks an incompatible change and triggers a major release.
4. `docs:`, `test:`, `chore:`, `refactor:`, `ci:`, `build:`, and `style:` do not trigger a release unless they are marked breaking.

Examples:

1. `feat(ui): add SDXL preset selector`
2. `fix(installer): resolve Blender 5.1 pip wheel selection`
3. `feat!: drop Blender 4.x support`
4. `docs(releasing): describe release bootstrap`

## SemVer Decision Guide

Use these questions to choose the right type:

1. Does the change break existing behavior, configuration, or supported Blender/Python versions?
   Use `!` and document the migration in a `BREAKING CHANGE:` footer when needed.
2. Does the change add new user-visible functionality or support a new workflow or integration?
   Use `feat:`.
3. Does the change fix a bug, packaging issue, installer issue, or broken runtime behavior?
   Use `fix:`.
4. Is the change only documentation, tests, CI, refactoring, or cleanup without behavior changes?
   Use the matching non-release type.

Repo-specific guidance:

1. Dropping Blender or Python support is major.
2. Adding a new addon capability, model family, or user-facing workflow is minor.
3. Correcting broken installer behavior, runtime handling, or packaging is patch.
4. Pure internal cleanup is not a release bump.

## Release Branch And Merge Style

`master` is the only release branch.

Prefer squash or rebase merges that preserve a Conventional Commit title. Non-conventional merge commit messages are valid Git history, but `release-please` only derives semver bumps from Conventional Commit style messages.

## Version Files

Do not manually bump `pyproject.toml` or `blender_manifest.toml` to cut a normal release. `release-please` owns version updates for those files and the changelog.
