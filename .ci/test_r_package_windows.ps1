# Download a file and retry upon failure. This looks like
# an infinite loop but CI-level timeouts will kill it
function Download-File-With-Retries {
  param(
    [string]$url,
    [string]$destfile
  )
  $ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
  do {
    Write-Output "Downloading ${url}"
    sleep 5;
    Invoke-WebRequest -Uri $url -OutFile $destfile
  } while(!$?);
}

# External utilities like R.exe / Rscript.exe writing to stderr (even for harmless
# status information) can cause failures in GitHub Actions PowerShell jobs.
# See https://github.community/t/powershell-steps-fail-nondeterministically/115496
#
# Using standard PowerShell redirection does not work to avoid these errors.
# This function uses R's built-in redirection mechanism, sink(). Any place where
# this function is used is a command that writes harmless messages to stderr
function Run-R-Code-Redirect-Stderr {
  param(
    [string]$rcode
  )
  $decorated_code = "out_file <- file(tempfile(), open = 'wt'); sink(out_file, type = 'message'); $rcode; sink()"
  Rscript --vanilla -e $decorated_code
}

# Remove all items matching some pattern from PATH environment variable
function Remove-From-Path {
  param(
    [string]$pattern_to_remove
  )
  $env:PATH = ($env:PATH.Split(';') | Where-Object { $_ -notmatch "$pattern_to_remove" }) -join ';'
}

# remove some details that exist in the GitHub Actions images which might
# cause conflicts with R and other components installed by this script
$env:RTOOLS40_HOME = ""
Remove-From-Path ".*Amazon.*"
Remove-From-Path ".*Anaconda.*"
Remove-From-Path ".*android.*"
Remove-From-Path ".*Android.*"
Remove-From-Path ".*chocolatey.*"
Remove-From-Path ".*Chocolatey.*"
Remove-From-Path ".*\\Git\\.*"
Remove-From-Path "(?!.*pandoc.*).*hostedtoolcache.*"
Remove-From-Path ".*Microsoft SDKs.*"
Remove-From-Path ".*mingw.*"
Remove-From-Path ".*msys64.*"
Remove-From-Path ".*PostgreSQL.*"
Remove-From-Path ".*\\R\\.*"
Remove-From-Path ".*R Client.*"
Remove-From-Path ".*rtools40.*"
Remove-From-Path ".*rtools42.*"
Remove-From-Path ".*rtools43.*"
Remove-From-Path ".*shells.*"
Remove-From-Path ".*Strawberry.*"
Remove-From-Path ".*tools.*"

Remove-Item C:\rtools40 -Force -Recurse -ErrorAction Ignore
Remove-Item C:\rtools42 -Force -Recurse -ErrorAction Ignore
Remove-Item C:\rtools43 -Force -Recurse -ErrorAction Ignore

# Get details needed for installing R components
#
# NOTES:
#    * some paths and file names are different on R4.0
$env:R_MAJOR_VERSION = $env:R_VERSION.split('.')[0]
if ($env:R_MAJOR_VERSION -eq "3") {
  # Rtools 3.x has to be installed at C:\Rtools\
  #     * https://stackoverflow.com/a/46619260/3986677
  $RTOOLS_INSTALL_PATH = "C:\Rtools"
  $env:RTOOLS_BIN = "$RTOOLS_INSTALL_PATH\bin"
  $env:RTOOLS_MINGW_BIN = "$RTOOLS_INSTALL_PATH\mingw_64\bin"
  $env:RTOOLS_EXE_FILE = "rtools35-x86_64.exe"
  $env:R_WINDOWS_VERSION = "3.6.3"
} elseif ($env:R_MAJOR_VERSION -eq "4") {
  $RTOOLS_INSTALL_PATH = "C:\rtools43"
  $env:RTOOLS_BIN = "$RTOOLS_INSTALL_PATH\usr\bin"
  $env:RTOOLS_MINGW_BIN = "$RTOOLS_INSTALL_PATH\x86_64-w64-mingw32.static.posix\bin"
  $env:RTOOLS_EXE_FILE = "rtools43-5550-5548.exe"
  $env:R_WINDOWS_VERSION = "4.3.1"
} else {
  Write-Output "[ERROR] Unrecognized R version: $env:R_VERSION"
  Check-Output $false
}

$env:R_LIB_PATH = "$env:BUILD_SOURCESDIRECTORY/RLibrary" -replace '[\\]', '/'
$env:R_LIBS = "$env:R_LIB_PATH"
$env:PATH = "$env:RTOOLS_BIN;" + "$env:RTOOLS_MINGW_BIN;" + "$env:R_LIB_PATH/R/bin/x64;"+ $env:PATH
if ([version]$env:R_VERSION -lt [version]"4.0") {
  $env:CRAN_MIRROR = "https://cran-archive.r-project.org"
} else {
  $env:CRAN_MIRROR = "https://cran.rstudio.com"
}
$env:MIKTEX_EXCEPTION_PATH = "$env:TEMP\miktex"

# don't fail builds for long-running examples unless they're very long.
# See https://github.com/microsoft/LightGBM/issues/4049#issuecomment-793412254.
if ($env:R_BUILD_TYPE -ne "cran") {
    $env:_R_CHECK_EXAMPLE_TIMING_THRESHOLD_ = 30
}

if (($env:COMPILER -eq "MINGW") -and ($env:R_BUILD_TYPE -eq "cmake")) {
  $env:CXX = "$env:RTOOLS_MINGW_BIN/g++.exe"
  $env:CC = "$env:RTOOLS_MINGW_BIN/gcc.exe"
}

cd $env:BUILD_SOURCESDIRECTORY
tzutil /s "GMT Standard Time"
[Void][System.IO.Directory]::CreateDirectory($env:R_LIB_PATH)

# download R and RTools
Write-Output "Downloading R and Rtools"
Download-File-With-Retries -url "$env:CRAN_MIRROR/bin/windows/base/old/$env:R_WINDOWS_VERSION/R-$env:R_WINDOWS_VERSION-win.exe" -destfile "R-win.exe"
Download-File-With-Retries -url "https://github.com/microsoft/LightGBM/releases/download/v2.0.12/$env:RTOOLS_EXE_FILE" -destfile "Rtools.exe"

# Install R
Write-Output "Installing R"
Start-Process -FilePath R-win.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /DIR=$env:R_LIB_PATH/R /COMPONENTS=main,x64,i386" ; Check-Output $?
Write-Output "Done installing R"

Write-Output "Installing Rtools"
Start-Process -FilePath Rtools.exe -NoNewWindow -Wait -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /DIR=$RTOOLS_INSTALL_PATH" ; Check-Output $?
Write-Output "Done installing Rtools"

Write-Output "Installing dependencies"
$packages = "c('data.table', 'jsonlite', 'knitr', 'markdown', 'Matrix', 'processx', 'R6', 'RhpcBLASctl', 'testthat'), dependencies = c('Imports', 'Depends', 'LinkingTo')"
Run-R-Code-Redirect-Stderr "options(install.packages.check.source = 'no'); install.packages($packages, repos = '$env:CRAN_MIRROR', type = 'binary', lib = '$env:R_LIB_PATH', Ncpus = parallel::detectCores())" ; Check-Output $?

Write-Output "Building R package"

# R CMD check is not used for MSVC builds
if ($env:COMPILER -ne "MSVC") {

  $PKG_FILE_NAME = "lightgbm_$env:LGB_VER.tar.gz"
  $LOG_FILE_NAME = "lightgbm.Rcheck/00check.log"

  if ($env:R_BUILD_TYPE -eq "cmake") {
    if ($env:TOOLCHAIN -eq "MINGW") {
      Write-Output "Telling R to use MinGW"
      $env:BUILD_R_FLAGS = "c('--skip-install', '--use-mingw', '-j4')"
    } elseif ($env:TOOLCHAIN -eq "MSYS") {
      Write-Output "Telling R to use MSYS"
      $env:BUILD_R_FLAGS = "c('--skip-install', '--use-msys2', '-j4')"
    } elseif ($env:TOOLCHAIN -eq "MSVC") {
      $env:BUILD_R_FLAGS = "'--skip-install'"
    } else {
      Write-Output "[ERROR] Unrecognized toolchain: $env:TOOLCHAIN"
      Check-Output $false
    }
    Run-R-Code-Redirect-Stderr "commandArgs <- function(...){$env:BUILD_R_FLAGS}; source('build_r.R')"; Check-Output $?
  } elseif ($env:R_BUILD_TYPE -eq "cran") {
    # NOTE: gzip and tar are needed to create a CRAN package on Windows, but
    # some flavors of tar.exe can fail in some settings on Windows.
    # Putting the msys64 utilities at the beginning of PATH temporarily to be
    # sure they're used for that purpose.
    if ($env:R_MAJOR_VERSION -eq "3") {
      $env:PATH = "C:\msys64\usr\bin;" + $env:PATH
    }
    Run-R-Code-Redirect-Stderr "result <- processx::run(command = 'sh', args = 'build-cran-package.sh', echo = TRUE, windows_verbatim_args = FALSE, error_on_status = TRUE)" ; Check-Output $?
    Remove-From-Path ".*msys64.*"
    # Test CRAN source .tar.gz in a directory that is not this repo or below it.
    # When people install.packages('lightgbm'), they won't have the LightGBM
    # git repo around. This is to protect against the use of relative paths
    # like ../../CMakeLists.txt that would only work if you are in the repoo
    $R_CMD_CHECK_DIR = "tmp-r-cmd-check"
    New-Item -Path "C:\" -Name $R_CMD_CHECK_DIR -ItemType "directory" > $null
    Move-Item -Path "$PKG_FILE_NAME" -Destination "C:\$R_CMD_CHECK_DIR\" > $null
    cd "C:\$R_CMD_CHECK_DIR\"
  }

  Write-Output "Running R CMD check"
  if ($env:R_BUILD_TYPE -eq "cran") {
    # CRAN packages must pass without --no-multiarch (build on 64-bit and 32-bit)
    $check_args = "c('CMD', 'check', '--as-cran', '--run-donttest', '$PKG_FILE_NAME')"
  } else {
    $check_args = "c('CMD', 'check', '--no-multiarch', '--as-cran', '--run-donttest', '$PKG_FILE_NAME')"
  }
  Run-R-Code-Redirect-Stderr "result <- processx::run(command = 'R.exe', args = $check_args, echo = TRUE, windows_verbatim_args = FALSE, error_on_status = TRUE)" ; $check_succeeded = $?

  Write-Output "R CMD check build logs:"
  $INSTALL_LOG_FILE_NAME = "lightgbm.Rcheck\00install.out"
  Get-Content -Path "$INSTALL_LOG_FILE_NAME"

  Check-Output $check_succeeded

  Write-Output "Looking for issues with R CMD check results"
  if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "NOTE|WARNING|ERROR" -CaseSensitive -Quiet) {
      echo "NOTEs, WARNINGs, or ERRORs have been found by R CMD check"
      Check-Output $False
  }

} else {
  $env:TMPDIR = $env:USERPROFILE  # to avoid warnings about incremental builds inside a temp directory
  $INSTALL_LOG_FILE_NAME = "$env:BUILD_SOURCESDIRECTORY\00install_out.txt"
  Run-R-Code-Redirect-Stderr "source('build_r.R')" 1> $INSTALL_LOG_FILE_NAME ; $install_succeeded = $?
  Write-Output "----- build and install logs -----"
  Get-Content -Path "$INSTALL_LOG_FILE_NAME"
  Write-Output "----- end of build and install logs -----"
  Check-Output $install_succeeded
  # some errors are not raised above, but can be found in the logs
  if (Get-Content "$INSTALL_LOG_FILE_NAME" | Select-String -Pattern "ERROR" -CaseSensitive -Quiet) {
      echo "ERRORs have been found installing lightgbm"
      Check-Output $False
  }
}

# Checking that the correct R version was used
if ($env:TOOLCHAIN -ne "MSVC") {
  $checks = Select-String -Path "${LOG_FILE_NAME}" -Pattern "using R version $env:R_WINDOWS_VERSION"
  $checks_cnt = $checks.Matches.length
} else {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "R version passed into FindLibR.* $env:R_WINDOWS_VERSION"
  $checks_cnt = $checks.Matches.length
}
if ($checks_cnt -eq 0) {
  Write-Output "Wrong R version was found (expected '$env:R_WINDOWS_VERSION'). Check the build logs."
  Check-Output $False
}

# Checking that we actually got the expected compiler. The R package has some logic
# to fail back to MinGW if MSVC fails, but for CI builds we need to check that the correct
# compiler was used.
if ($env:R_BUILD_TYPE -eq "cmake") {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "Check for working CXX compiler.*$env:COMPILER"
  if ($checks.Matches.length -eq 0) {
    Write-Output "The wrong compiler was used. Check the build logs."
    Check-Output $False
  }
}

# Checking that we got the right toolchain for MinGW. If using MinGW, both
# MinGW and MSYS toolchains are supported
if (($env:COMPILER -eq "MINGW") -and ($env:R_BUILD_TYPE -eq "cmake")) {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "Trying to build with.*$env:TOOLCHAIN"
  if ($checks.Matches.length -eq 0) {
    Write-Output "The wrong toolchain was used. Check the build logs."
    Check-Output $False
  }
}

# Checking that MM_PREFETCH preprocessor definition is actually used in CI builds.
if ($env:R_BUILD_TYPE -eq "cran") {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "checking whether MM_PREFETCH work.*yes"
  $checks_cnt = $checks.Matches.length
} elseif ($env:TOOLCHAIN -ne "MSVC") {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern ".*Performing Test MM_PREFETCH - Success"
  $checks_cnt = $checks.Matches.length
} else {
  $checks_cnt = 1
}
if ($checks_cnt -eq 0) {
  Write-Output "MM_PREFETCH preprocessor definition wasn't used. Check the build logs."
  Check-Output $False
}

# Checking that MM_MALLOC preprocessor definition is actually used in CI builds.
if ($env:R_BUILD_TYPE -eq "cran") {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "checking whether MM_MALLOC work.*yes"
  $checks_cnt = $checks.Matches.length
} elseif ($env:TOOLCHAIN -ne "MSVC") {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern ".*Performing Test MM_MALLOC - Success"
  $checks_cnt = $checks.Matches.length
} else {
  $checks_cnt = 1
}
if ($checks_cnt -eq 0) {
  Write-Output "MM_MALLOC preprocessor definition wasn't used. Check the build logs."
  Check-Output $False
}

# Checking that OpenMP is actually used in CMake builds.
if ($env:R_BUILD_TYPE -eq "cmake") {
  $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern ".*Found OpenMP: TRUE.*"
  if ($checks.Matches.length -eq 0) {
    Write-Output "OpenMP wasn't found. Check the build logs."
    Check-Output $False
  }
}

if ($env:COMPILER -eq "MSVC") {
  Write-Output "Running tests with testthat.R"
  cd R-package/tests
  # NOTE: using Rscript.exe intentionally here, instead of Run-R-Code-Redirect-Stderr,
  #       because something about the interaction between Run-R-Code-Redirect-Stderr
  #       and testthat results in failing tests not exiting with a non-0 exit code.
  Rscript.exe --vanilla "testthat.R" ; Check-Output $?
}

Write-Output "No issues were found checking the R package"
