# Releasing

## Overview

`master` is the only release branch. Releases are automated by `.github/workflows/release-please.yml` and `release-please-config.json`.

On every push to `master`, the workflow:

1. Parses the Conventional Commits since the last release.
2. Opens or updates a release PR when a releasable change (`feat:`, `fix:` or a breaking change) is present.
3. When that release PR is merged: creates the tag and GitHub release, builds `diffused_texture_addon-<version>.zip` from the released commit, and uploads it to the release.

Tags look like `diffused_texture_addon-v<version>`.

## Normal Release Flow

1. Merge Conventional Commit changes into `master` (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
2. Let `release-please` keep the release PR up to date as more changes land.
3. Merge the release PR when you want to publish.

Do not bump `pyproject.toml`, `blender_manifest.toml` or `CHANGELOG.md` by hand; the release PR does that.

To force a specific version, run the `release-please` workflow manually with the `release_as` input.

## CI/CD Layout

1. `tests.yml` (Tests and lint): ruff, the fast test suite and the installer contract tests on every push and pull request.
2. `artifact-smoke.yml`: builds the ZIP and smoke-tests it on both Blender/Python lanes (push, pull request and nightly).
3. `build.yml`: manual preview build, also callable from other workflows.
4. `release-please.yml`: release PRs, tags, changelog, GitHub releases and release assets.

## Why `RELEASE_PLEASE_TOKEN` Matters

The workflow falls back to the default `GITHUB_TOKEN`, but GitHub does not trigger follow-up workflows for pull requests and tags created with that token. With a `RELEASE_PLEASE_TOKEN` secret (a token that can write contents, pull requests and issues), the release PR gets the same CI checks as any other PR.
