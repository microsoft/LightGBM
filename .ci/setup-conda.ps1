function Assert-Output {
    param( [Parameter(Mandatory = $true)][bool]$success )
    if (-not $success) {
        $host.SetShouldExit(-1)
        exit 1
    }
}

Write-Output "Downloading miniforge installer"
$ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down Invoke-WebRequest
$miniforgeInstaller = "$env:TEMP\Miniforge3.exe"
Invoke-WebRequest `
    -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" `
    -OutFile $miniforgeInstaller

Write-Output "Installing conda with miniforge"
Start-Process -FilePath $miniforgeInstaller -Wait -NoNewWindow -ArgumentList @(
    "/S",
    "/AddToPath=1",
    "/InstallationType=JustMe",
    "/RegisterPython=0",
    "/D=C:\Miniforge3"
) ; Assert-Output $?
Remove-Item $miniforgeInstaller

# ensure miniforge is at the beginning of PATH
$env:PATH = @(
    "C:\Miniforge3\Scripts",
    "C:\Miniforge3\condabin",
    $env:PATH
) -join ";"
Write-Output "Done installing conda with miniforge"
