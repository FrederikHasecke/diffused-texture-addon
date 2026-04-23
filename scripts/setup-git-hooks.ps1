$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    git config core.hooksPath ".githooks"
    $configuredPath = git config --local --get core.hooksPath
    Write-Host "Configured core.hooksPath=$configuredPath"
    Write-Host "Git will now run the semver prompt and Conventional Commit validator on local commits."
}
finally {
    Pop-Location
}
