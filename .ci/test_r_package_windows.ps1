# Download a file and retry upon failure. This looks like
# an infinite looop 
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
Add-Content .Rprofile "options(repos = 'https://cran.rstudio.com')"
Add-Content .Rprofile "options(pkgType = 'binary')"
Add-Content .Rprofile "options(install.packages.check.source = 'no')"

Write-Output "Installing dependencies"
Rscript.exe -e "install.packages(c('data.table', 'jsonlite', 'Matrix', 'R6', 'testthat'), dependencies = c('Imports', 'Depends', 'LinkingTo'), lib = '$env:R_LIB_PATH')" ; Check-Output $?

Write-Output "Building R package"
Rscript build_r.R ; Check-Output $?

$PKG_FILE_NAME = Get-Item *.tar.gz
$PKG_NAME = $PKG_FILE_NAME.BaseName.split("_")[0]
$LOG_FILE_NAME = "$PKG_NAME.Rcheck/00check.log"

Write-Output "Running R CMD check"
$env:_R_CHECK_FORCE_SUGGESTS_=0
R.exe CMD check "${PKG_FILE_NAME}" --as-cran --no-multiarch; Check-Output $?

if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "WARNING" -Quiet) {
    echo "WARNINGS have been found by R CMD check!"
    Check-Output $False
}

Exit 0
