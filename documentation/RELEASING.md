# Releasing

## Overview

`master` is the release branch.

Release automation is handled by `.github/workflows/release-please.yml` plus `release-please-config.json`.

On every push to `master`, the workflow:

1. Parses Conventional Commits with `release-please`.
2. Opens or updates a release PR when a releasable change is present.
3. Creates a real GitHub release when that release PR is merged.
4. Builds `diffused_texture_addon-<version>.zip` from the released SHA.
5. Uploads the zip to the GitHub release.

## First Bootstrap On `master`

The latest tagged release on `master` is `v0.1.0`, while newer work has already moved the working version files forward. Older history also is not consistently written as Conventional Commits.

Use this one-time bootstrap sequence after these automation changes land on `master`:

1. Merge this automation to `master`.
2. Create a repository secret named `RELEASE_PLEASE_TOKEN` with a GitHub token that can write contents, pull requests, and issues.
3. In repository settings, enable `Allow GitHub Actions to create and approve pull requests`.
4. Run the `release-please` workflow manually with `release_as=0.2.0`.
5. Review the generated release PR.
6. Merge the release PR into `master`.

After that bootstrap release, normal Conventional Commits on `master` drive future versions automatically.

## Normal Release Flow

1. Merge Conventional Commit based changes into `master`.
2. Let `release-please` update the release PR as more releasable changes land.
3. Merge the release PR when you want to publish.
4. The workflow immediately publishes the real GitHub release and uploads the addon zip.

## CI/CD Layout

1. `.github/workflows/lint.yml` is the normal CI workflow for pushes and pull requests.
2. `.github/workflows/build.yml` is now a manual preview build workflow.
3. `.github/workflows/release-please.yml` owns release PRs, tags, changelog updates, GitHub releases, and release assets.

## Why `RELEASE_PLEASE_TOKEN` Matters

The workflow falls back to the default `GITHUB_TOKEN`, but GitHub does not trigger follow-up workflows from pull requests and tags created by that token.

Using `RELEASE_PLEASE_TOKEN` is strongly recommended so the release PR behaves like a normal PR and receives the same CI feedback.
