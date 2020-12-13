
$cache = "$env:PIPELINE_WORKSPACE\opencl_windows-amd_cpu-v3_0_130_135"
$installer = "AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe"

Write-Output "Downloading OpenCL platform installer"
$parts = @("1", "2", "3", "4", "5", "6", "7", "8", "9", "EXE")
foreach ($p in $parts) {
  Write-Output " - downloading part $($p)"
  Invoke-WebRequest -OutFile "$installer.$p" -Uri "https://gamma-rho.com/parts/$installer.$p"
}

Write-Output "Combining OpenCL platform installer parts"
Start-Process "$installer.EXE" -Wait
Start-Sleep -Seconds 10
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

