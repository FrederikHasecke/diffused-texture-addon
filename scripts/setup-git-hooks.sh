#!/bin/sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

chmod +x .githooks/run-hook-helper .githooks/pre-commit .githooks/prepare-commit-msg .githooks/commit-msg
git config core.hooksPath .githooks

printf '%s\n' "Configured core.hooksPath=$(git config --local --get core.hooksPath)"
printf '%s\n' 'Git will now run the semver prompt and Conventional Commit validator on local commits.'
