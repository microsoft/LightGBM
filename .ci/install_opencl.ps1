
Write-Output "Installing OpenCL CPU platform"

Write-Output "Agent platform information:"
Get-WmiObject -Class Win32_ComputerSystem
Get-WmiObject -Class Win32_Processor
Get-WmiObject -Class Win32_BIOS

$cache = "$env:PIPELINE_WORKSPACE\opencl_windows-amd_cpu-v3_0_130_135"
$installer = "AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe"
Write-Output "Running OpenCL installer"
Invoke-Command -ScriptBlock {Start-Process "$cache\$installer" -ArgumentList '/S /V"/quiet /norestart /passive /log opencl.log"' -Wait}

$property = Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors
if ($property -eq $null) {
  Write-Output "Unable to install OpenCL CPU platform"
  Write-Output "Setting EXIT"
  $host.SetShouldExit(-1)
  Exit -1
} else {
  Write-Output "Successfully installed OpenCL CPU platform"
  Write-Output "Current OpenCL drivers:"
  Write-Output $property
}

