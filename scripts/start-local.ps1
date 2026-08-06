param(
    [switch]$EnableExecution,
    [switch]$EnableTailnetAccess,
    [switch]$Development,
    [string]$WslDistro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot "backend\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Backend environment is missing. Run scripts/setup.ps1 first."
}

$previousExecution = $env:AUTORESEARCH_ENABLE_EXECUTION
$previousWsl = $env:AUTORESEARCH_WSL_DISTRO
$env:AUTORESEARCH_ENABLE_EXECUTION = if ($EnableExecution) { "1" } else { "0" }
$env:AUTORESEARCH_WSL_DISTRO = $WslDistro

$backendProcess = $null
Push-Location $projectRoot
try {
    $backendProcess = Start-Process `
        -FilePath $venvPython `
        -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "7331") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru
    if ($EnableTailnetAccess) {
        $tailscale = Get-Command tailscale -ErrorAction SilentlyContinue
        if (-not $tailscale) {
            throw "Tailscale is not installed. Install and sign in, then run again with -EnableTailnetAccess."
        }
        & $tailscale.Source serve --bg http://localhost:3000
        & $tailscale.Source serve --bg --set-path /api http://localhost:7331/api
        & $tailscale.Source serve --bg --set-path /health http://localhost:7331/health
        Write-Host "Private Mac access is enabled through Tailscale Serve."
        & $tailscale.Source serve status
    }
    if ($Development) {
        npm run dev
    }
    else {
        $productionEntry = Join-Path $projectRoot "dist\server\index.js"
        if (-not (Test-Path -LiteralPath $productionEntry)) {
            Write-Host "Building the ResearchLab web client for production."
            npm run build
            if ($LASTEXITCODE -ne 0) {
                throw "ResearchLab production build failed with exit code $LASTEXITCODE."
            }
        }
        npm run start
    }
}
finally {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id
    }
    $env:AUTORESEARCH_ENABLE_EXECUTION = $previousExecution
    $env:AUTORESEARCH_WSL_DISTRO = $previousWsl
    Pop-Location
}
