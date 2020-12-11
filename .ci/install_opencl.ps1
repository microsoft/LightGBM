
Write-Output "Agent platform information:"
Get-WmiObject -Class Win32_ComputerSystem
Get-WmiObject -Class Win32_Processor
Get-WmiObject -Class Win32_BIOS

if ($env:TASK -eq "bdist") {
  Write-Output "Downloading OpenCL runtime"
  $installer = "AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe"
  curl -o .\$installer https://gamma-rho.com/$installer

  Write-Output "Installing OpenCL runtime"
  Invoke-Command -ScriptBlock {Start-Process .\$installer -ArgumentList '/S /V"/quiet /norestart /passive /log amd_opencl_sdk.log"' -Wait}

  Write-Output "Current OpenCL drivers:"
  Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors
}

