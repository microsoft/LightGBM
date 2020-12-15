
Write-Output "Installing OpenCL CPU platform"

Write-Output "Agent platform information:"
Get-WmiObject -Class Win32_ComputerSystem
Get-WmiObject -Class Win32_Processor
Get-WmiObject -Class Win32_BIOS

$installer = "AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe"

Write-Output "Downloading OpenCL installer"
Invoke-WebRequest -OutFile "$installer" -Uri "https://github.com/microsoft/LightGBM/releases/download/v2.0.12/$installer"
if (-Not (Test-Path ".\$installer")) {
  Write-Output "Unable to download OpenCL installer"
  Write-Output "Unable to install OpenCL CPU platform"
  Write-Output "Setting EXIT"
  $host.SetShouldExit(-1)
  Exit -1
}

Write-Output "Running OpenCL installer"
Invoke-Command -ScriptBlock {Start-Process "$installer" -ArgumentList '/S /V"/quiet /norestart /passive /log opencl.log"' -Wait}
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

