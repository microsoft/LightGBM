# [description]
#
#   Installs a native win-arm64 CPython on the 'windows-11-arm' CI runner.
#
#   That runner's pre-installed Miniconda targets x64 (conda-forge / Miniforge don't yet
#   support win-arm64: https://conda-forge.org/blog/2026/02/09/win-arm64/) and would run
#   under emulation, and 'actions/setup-python' does not yet resolve an arm64 interpreter
#   on Windows (https://github.com/actions/setup-python/issues/976). python.org has
#   published native win-arm64 installers since Python 3.11, so install directly from there.
#
# [usage]
#
#   pwsh -File .ci/install-python-arm64-windows.ps1

function Assert-Output {
    param( [Parameter(Mandatory = $true)][bool]$success )
    if (-not $success) {
        $host.SetShouldExit(-1)
        exit 1
    }
}

$PythonVersion = "3.13.1"
$InstallDir = "$env:USERPROFILE\python-arm64"
$Installer = "python-$PythonVersion-arm64.exe"
$Uri = "https://www.python.org/ftp/python/$PythonVersion/$Installer"

Write-Output "Downloading $Uri"
$ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
Invoke-WebRequest -Uri $Uri -OutFile $Installer ; Assert-Output $?

Write-Output "Installing Python $PythonVersion (arm64) to $InstallDir"
$installArgs = @(
    "/quiet",
    "InstallAllUsers=0",
    "PrependPath=0",
    "Include_launcher=0",
    "Include_test=0",
    "TargetDir=$InstallDir"
)
Start-Process -FilePath ".\$Installer" -ArgumentList $installArgs -Wait

if (-not (Test-Path "$InstallDir\python.exe")) {
    Write-Output "Python installation failed"
    $host.SetShouldExit(-1)
    exit 1
}

# make this Python (and 'pip'-installed console scripts) available to later steps in this job
Add-Content -Path $env:GITHUB_PATH -Value $InstallDir
Add-Content -Path $env:GITHUB_PATH -Value "$InstallDir\Scripts"
