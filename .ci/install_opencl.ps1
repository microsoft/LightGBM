
Write-Output "Installing OpenCL CPU platform"

$cache = "$env:PIPELINE_WORKSPACE\opencl_windows-amd_cpu-v3_0_130_135"
$installer = "AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe"

if ($env:OPENCL_INSTALLER_FOUND -ne 'true') {
  # Pipeline cache miss; download OpenCL platform installer executable into workspace cache

  Write-Output "Downloading OpenCL platform installer"
  Invoke-WebRequest -OutFile "$installer" -Uri "https://github.com/microsoft/LightGBM/releases/download/v2.0.12/$installer"

  Write-Output "Caching OpenCL platform installer"
  New-Item $cache -ItemType Directory | Out-Null
  Move-Item -Path "$installer" -Destination "$cache\$installer" | Out-Null

  if (Test-Path "$cache\$installer") {
    Write-Output "Successfully downloaded OpenCL platform installer"
  } else {
    Write-Output "Unable to download OpenCL platform installer"
    Write-Output "Setting EXIT"
    $host.SetShouldExit(-1)
    Exit -1
  }
}

# Install OpenCL platform from installer executable expected in workspace cache

Write-Output "Running OpenCL installer"
Invoke-Command -ScriptBlock {Start-Process "$cache\$installer" -ArgumentList '/S /V"/quiet /norestart /passive /log opencl.log"' -Wait}

$property = Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors
if ($property -eq $null) {
  Write-Output "Unable to install OpenCL CPU platform"
  Write-Output "OpenCL installation log:"
  Get-Content "opencl.log"
  Write-Output "Setting EXIT"
  $host.SetShouldExit(-1)
  Exit -1
} else {
  Write-Output "Successfully installed OpenCL CPU platform"
  Write-Output "Current OpenCL drivers:"
  Write-Output $property
}

