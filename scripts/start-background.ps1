$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\saita\Documents\Researching'
$logDir = Join-Path $repo 'backend\data\logs\service'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$created = $false
$mutex = New-Object System.Threading.Mutex($true, 'Local\ResearchLabBackground', [ref]$created)
if (-not $created) { exit 0 }
$logFile = Join-Path $logDir ('ResearchLab-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
try {
    [Environment]::SetEnvironmentVariable('__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS', 'ibrahim.tailae27bb.ts.net', 'User')
    $env:__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS = 'ibrahim.tailae27bb.ts.net'
    & (Join-Path $repo 'scripts\start-local.ps1') -EnableExecution -EnableTailnetAccess *>> $logFile
    exit $LASTEXITCODE
}
catch {
    $_ | Out-File -FilePath $logFile -Append
    exit 1
}
finally {
    if ($created) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
