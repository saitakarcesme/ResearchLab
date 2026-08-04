$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot "backend\.venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    npm install
    if (-not (Test-Path -LiteralPath $venvPython)) {
        py -3.11 -m venv "backend\.venv"
    }
    & $venvPython -m pip install --disable-pip-version-check -r "backend\requirements.txt"
}
finally {
    Pop-Location
}
