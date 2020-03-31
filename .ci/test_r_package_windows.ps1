# Download a file and retry upon failure. This looks like
# an infinite loop but CI-level timeouts will kill it
function Download-File-With-Retries {
  param(
    [string]$url,
    [string]$destfile
  )
  do {
    Write-Output "Downloading '${url}'"
    sleep 5;
    (New-Object System.Net.WebClient).DownloadFile($url, $destfile)
  } while(!$?);
}

$env:R_LIB_PATH = "C:/RLibrary"
$env:PATH = "$env:R_LIB_PATH/Rtools/bin;" + "$env:R_LIB_PATH/R/bin/x64;" + "$env:R_LIB_PATH/miktex/texmfs/install/miktex/bin/x64;" + $env:PATH
$env:BINPREF = "C:/mingw-w64/x86_64-8.1.0-posix-seh-rt_v6-rev0/mingw64/bin/"
$env:CRAN_MIRROR = "https://cloud.r-project.org/"

cd $env:BUILD_SOURCESDIRECTORY
tzutil /s "GMT Standard Time"
[Void][System.IO.Directory]::CreateDirectory($env:R_LIB_PATH)

if ($env:COMPILER -eq "MINGW") {
  Write-Output "Telling R to use MinGW"
  $install_libs = "$env:BUILD_SOURCESDIRECTORY\R-package\src\install.libs.R"
  ((Get-Content -path $install_libs -Raw) -replace 'use_mingw <- FALSE','use_mingw <- TRUE') | Set-Content -Path $install_libs
}

# set up R if it doesn't exist yet
if (!(Get-Command R.exe -errorAction SilentlyContinue)) {

    Write-Output "Downloading R and Rtools"

    # download R and RTools
    Download-File-With-Retries -url "https://cloud.r-project.org/bin/windows/base/old/$env:R_WINDOWS_VERSION/R-$env:R_WINDOWS_VERSION-win.exe" -destfile "R-win.exe"
    Download-File-With-Retries -url "https://cloud.r-project.org/bin/windows/Rtools/Rtools35.exe" -destfile "Rtools.exe"

    # Install R
    Write-Output "Installing R"
    Start-Process -FilePath R-win.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH/R /COMPONENTS=main,x64" ; Check-Output $?
    Write-Output "Done installing R"

    Write-Output "Installing Rtools"
    Start-Process -FilePath Rtools.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH/Rtools" ; Check-Output $?
    Write-Output "Done installing Rtools"

    # download Miktex
    Write-Output "Downloading MiKTeX"
    Download-File-With-Retries -url "https://miktex.org/download/win/miktexsetup-x64.zip" -destfile "miktexsetup-x64.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory("miktexsetup-x64.zip", "miktex")
    Write-Output "Setting up MiKTeX"
    .\miktex\miktexsetup.exe --local-package-repository=./miktex/download --package-set=essential --quiet download ; Check-Output $?
    Write-Output "Installing MiKTeX"
    .\miktex\download\miktexsetup.exe --portable="$env:R_LIB_PATH/miktex" --quiet install ; Check-Output $?
    Write-Output "Done installing R, Rtools, and MiKTeX"
}

initexmf --set-config-value [MPM]AutoInstall=1
conda install -y --no-deps pandoc

Add-Content .Renviron "R_LIBS=$env:R_LIB_PATH"

Write-Output "Installing dependencies"
$packages = "c('data.table', 'jsonlite', 'Matrix', 'R6', 'testthat'), dependencies = c('Imports', 'Depends', 'LinkingTo')"
Rscript --vanilla -e "install.packages($packages, repos = '$env:CRAN_MIRROR', pkgType = 'binary', lib = '$env:R_LIB_PATH', install.packages.check.source = 'no')" ; Check-Output $?

Write-Output "Building R package"
Rscript build_r.R --skip-install ; Check-Output $?

$PKG_FILE_NAME = Get-Item *.tar.gz
$LOG_FILE_NAME = "lightgbm.Rcheck/00check.log"

$env:_R_CHECK_FORCE_SUGGESTS_=0
if ($env:AZURE -eq "true") {
  Write-Output "Running R CMD check without checking documentation"
  R.exe CMD check --no-multiarch --no-manual --ignore-vignettes ${PKG_FILE_NAME} ; Check-Output $?
} else {
  Write-Output "Running R CMD check as CRAN"
  R.exe CMD check --no-multiarch --as-cran ${PKG_FILE_NAME} ; Check-Output $?
}

Write-Output "Looking for issues with R CMD check results"
if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "WARNING" -Quiet) {
    echo "WARNINGS have been found by R CMD check!"
    Check-Output $False
}

Write-Output "No issues were found checking the R package"
Exit 0
