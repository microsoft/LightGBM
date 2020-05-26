# Download a file and retry upon failure. This looks like
# an infinite loop but CI-level timeouts will kill it
function Download-File-With-Retries {
  param(
    [string]$url,
    [string]$destfile
  )
  do {
    Write-Output "Downloading ${url}"
    sleep 5;
    (New-Object System.Net.WebClient).DownloadFile($url, $destfile)
  } while(!$?);
}

$env:R_WINDOWS_VERSION = "3.6.3"
$env:R_LIB_PATH = "$env:BUILD_SOURCESDIRECTORY/RLibrary" -replace '[\\]', '/'
$env:R_LIBS = "$env:R_LIB_PATH"
$env:PATH = "$env:R_LIB_PATH/Rtools/bin;" + "$env:R_LIB_PATH/R/bin/x64;" + "$env:R_LIB_PATH/miktex/texmfs/install/miktex/bin/x64;" + $env:PATH
$env:CRAN_MIRROR = "https://cloud.r-project.org/"
$env:CTAN_MIRROR = "https://ctan.math.illinois.edu/systems/win32/miktex/tm/packages/"

if ($env:COMPILER -eq "MINGW") {
  $env:CXX = "$env:R_LIB_PATH/Rtools/mingw_64/bin/g++.exe"
  $env:CC = "$env:R_LIB_PATH/Rtools/mingw_64/bin/gcc.exe"
}

cd $env:BUILD_SOURCESDIRECTORY
tzutil /s "GMT Standard Time"
[Void][System.IO.Directory]::CreateDirectory($env:R_LIB_PATH)

if ($env:COMPILER -eq "MINGW") {
  Write-Output "Telling R to use MinGW"
  $install_libs = "$env:BUILD_SOURCESDIRECTORY/R-package/src/install.libs.R"
  ((Get-Content -path $install_libs -Raw) -replace 'use_mingw <- FALSE','use_mingw <- TRUE') | Set-Content -Path $install_libs
}

# download R and RTools
Write-Output "Downloading R and Rtools"
Download-File-With-Retries -url "https://cloud.r-project.org/bin/windows/base/old/$env:R_WINDOWS_VERSION/R-$env:R_WINDOWS_VERSION-win.exe" -destfile "R-win.exe"
Download-File-With-Retries -url "https://cloud.r-project.org/bin/windows/Rtools/Rtools35.exe" -destfile "Rtools.exe"

# Install R
Write-Output "Installing R"
Start-Process -FilePath R-win.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH/R /COMPONENTS=main,x64" ; Check-Output $?
Write-Output "Done installing R"

Write-Output "Installing Rtools"
Start-Process -FilePath Rtools.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH/Rtools" ; Check-Output $?
Write-Output "Done installing Rtools"

Write-Output "Installing dependencies"
$packages = "c('data.table', 'jsonlite', 'httr', 'Matrix', 'processx', 'R6', 'testthat'), dependencies = c('Imports', 'Depends', 'LinkingTo')"
Rscript --vanilla -e "options(install.packages.check.source = 'no'); install.packages($packages, repos = '$env:CRAN_MIRROR', type = 'binary', lib = '$env:R_LIB_PATH')" | Out-String -Stream ; Check-Output $?

# MiKTeX and pandoc can be skipped on non-MINGW builds, since we don't
# build the package documentation for those
if ($env:COMPILER -eq "MINGW") {
    Write-Output "Downloading MiKTeX"
    Rscript $env:BUILD_SOURCESDIRECTORY\.ci\download-miktex.R "miktexsetup-x64.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory("miktexsetup-x64.zip", "miktex")
    Write-Output "Setting up MiKTeX"
    .\miktex\miktexsetup.exe --remote-package-repository="$env:CTAN_MIRROR" --local-package-repository=./miktex/download --package-set=essential --quiet download ; Check-Output $?
    Write-Output "Installing MiKTeX"
    .\miktex\download\miktexsetup.exe --remote-package-repository="$env:CTAN_MIRROR" --portable="$env:R_LIB_PATH/miktex" --quiet install ; Check-Output $?
    Write-Output "Done installing MiKTeX"

    initexmf --set-config-value [MPM]AutoInstall=1 | Out-String -Stream
    conda install -q -y --no-deps pandoc
}

Write-Output "Building R package"

# R CMD check is not used for MSVC builds
if ($env:COMPILER -ne "MSVC") {
  Rscript build_r.R --skip-install ; Check-Output $?

  $PKG_FILE_NAME = Get-Item *.tar.gz
  $LOG_FILE_NAME = "lightgbm.Rcheck/00check.log"

  $env:_R_CHECK_FORCE_SUGGESTS_ = 0
  Write-Output "Running R CMD check as CRAN"
  R.exe CMD check --no-multiarch --as-cran ${PKG_FILE_NAME} | Out-String -Stream ; $check_succeeded = $?

  Write-Output "R CMD check build logs:"
  $INSTALL_LOG_FILE_NAME = "$env:BUILD_SOURCESDIRECTORY\lightgbm.Rcheck\00install.out"
  Get-Content -Path "$INSTALL_LOG_FILE_NAME"

  Check-Output $check_succeeded

  Write-Output "Looking for issues with R CMD check results"
  if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "WARNING" -Quiet) {
      echo "WARNINGS have been found by R CMD check!"
      Check-Output $False
  }

  $note_str = Get-Content -Path "${LOG_FILE_NAME}" | Select-String -Pattern '.*Status.* NOTE' | Out-String ; Check-Output $?
  $relevant_line = $note_str -match '(\d+) NOTE'
  $NUM_CHECK_NOTES = $matches[1]
  $ALLOWED_CHECK_NOTES = 3
  if ([int]$NUM_CHECK_NOTES -gt $ALLOWED_CHECK_NOTES) {
      Write-Output "Found ${NUM_CHECK_NOTES} NOTEs from R CMD check. Only ${ALLOWED_CHECK_NOTES} are allowed"
      Check-Output $False
  }
} else {
  $INSTALL_LOG_FILE_NAME = "$env:BUILD_SOURCESDIRECTORY\00install_out.txt"
  Rscript build_r.R *> $INSTALL_LOG_FILE_NAME ; $install_succeeded = $?
  Write-Output "----- build and install logs -----"
  Get-Content -Path "$INSTALL_LOG_FILE_NAME"
  Write-Output "----- end of build and install logs -----"
  Check-Output $install_succeeded
}

# Checking that we actually got the expected compiler. The R package has some logic
# to fail back to MinGW if MSVC fails, but for CI builds we need to check that the correct
# compiler was used.
$checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "Check for working CXX compiler.*$env:COMPILER"
if ($checks.Matches.length -eq 0) {
  Write-Output "The wrong compiler was used. Check the build logs."
  Check-Output $False
}

if ($env:COMPILER -eq "MSVC") {
  Write-Output "Running tests with testthat.R"
  cd R-package/tests
  Rscript testthat.R ; Check-Output $?
}

Write-Output "No issues were found checking the R package"
