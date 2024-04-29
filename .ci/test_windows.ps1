function Check-Output {
  param( [bool]$success )
  if (!$success) {
    $host.SetShouldExit(-1)
    exit 1
  }
}

# unify environment variable for Azure DevOps and AppVeyor
if (Test-Path env:APPVEYOR) {
  $env:APPVEYOR = "true"
  $env:ALLOW_SKIP_ARROW_TESTS = "1"
}

if ($env:TASK -eq "r-package") {
  & $env:BUILD_SOURCESDIRECTORY\.ci\test_r_package_windows.ps1 ; Check-Output $?
  Exit 0
}

if ($env:TASK -eq "cpp-tests") {
  cmake -B build -S . -DBUILD_CPP_TEST=ON -DUSE_OPENMP=OFF -DUSE_DEBUG=ON -A x64
  cmake --build build --target testlightgbm --config Debug ; Check-Output $?
  .\Debug\testlightgbm.exe ; Check-Output $?
  Exit 0
}

if ($env:TASK -eq "swig") {
  $env:JAVA_HOME = $env:JAVA_HOME_8_X64  # there is pre-installed Eclipse Temurin 8 somewhere
  $ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
  Invoke-WebRequest -Uri "https://github.com/microsoft/LightGBM/releases/download/v2.0.12/swigwin-4.0.2.zip" -OutFile $env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip -UserAgent "NativeHost"
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory("$env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip", "$env:BUILD_SOURCESDIRECTORY/swig")
  $env:PATH = "$env:BUILD_SOURCESDIRECTORY/swig/swigwin-4.0.2;" + $env:PATH
  cmake -B build -S . -A x64 -DUSE_SWIG=ON ; Check-Output $?
  cmake --build build --target ALL_BUILD --config Release ; Check-Output $?
  if ($env:AZURE -eq "true") {
    cp $env:BUILD_SOURCESDIRECTORY/build/lightgbmlib.jar $env:BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_win.jar ; Check-Output $?
  }
  Exit 0
}

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no

# ref:
# * https://stackoverflow.com/a/62897729/3986677
# * https://github.com/microsoft/LightGBM/issues/5899
conda install "brotlipy>=0.7"

conda update -q -y conda

if ($env:PYTHON_VERSION -eq "3.7") {
  $env:CONDA_REQUIREMENT_FILE = "$env:BUILD_SOURCESDIRECTORY/.ci/conda-envs/ci-core-py37.txt"
} else {
  $env:CONDA_REQUIREMENT_FILE = "$env:BUILD_SOURCESDIRECTORY/.ci/conda-envs/ci-core.txt"
}

conda create `
  -y `
  -n $env:CONDA_ENV `
  --file $env:CONDA_REQUIREMENT_FILE `
  "python=$env:PYTHON_VERSION[build=*cpython]" ; Check-Output $?

if ($env:TASK -ne "bdist") {
  conda activate $env:CONDA_ENV
}

if ($env:TASK -eq "regular") {
  cmake -B build -S . -A x64 ; Check-Output $?
  cmake --build build --target ALL_BUILD --config Release ; Check-Output $?
  cd $env:BUILD_SOURCESDIRECTORY
  sh $env:BUILD_SOURCESDIRECTORY/build-python.sh install --precompile ; Check-Output $?
  cp $env:BUILD_SOURCESDIRECTORY/Release/lib_lightgbm.dll $env:BUILD_ARTIFACTSTAGINGDIRECTORY
  cp $env:BUILD_SOURCESDIRECTORY/Release/lightgbm.exe $env:BUILD_ARTIFACTSTAGINGDIRECTORY
}
elseif ($env:TASK -eq "sdist") {
  cd $env:BUILD_SOURCESDIRECTORY
  sh $env:BUILD_SOURCESDIRECTORY/build-python.sh sdist ; Check-Output $?
  sh $env:BUILD_SOURCESDIRECTORY/.ci/check_python_dists.sh $env:BUILD_SOURCESDIRECTORY/dist ; Check-Output $?
  cd dist; pip install @(Get-ChildItem *.gz) -v ; Check-Output $?
}
elseif ($env:TASK -eq "bdist") {
  # Import the Chocolatey profile module so that the RefreshEnv command
  # invoked below properly updates the current PowerShell session environment.
  $module = "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1"
  Import-Module "$module" ; Check-Output $?
  RefreshEnv

  Write-Output "Current OpenCL drivers:"
  Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors

  conda activate $env:CONDA_ENV
  cd $env:BUILD_SOURCESDIRECTORY
  sh "build-python.sh" bdist_wheel --integrated-opencl ; Check-Output $?
  sh $env:BUILD_SOURCESDIRECTORY/.ci/check_python_dists.sh $env:BUILD_SOURCESDIRECTORY/dist ; Check-Output $?
  cd dist; pip install --user @(Get-ChildItem *py3-none-win_amd64.whl) ; Check-Output $?
  cp @(Get-ChildItem *py3-none-win_amd64.whl) $env:BUILD_ARTIFACTSTAGINGDIRECTORY
} elseif (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python")) {
  cd $env:BUILD_SOURCESDIRECTORY
  if ($env:COMPILER -eq "MINGW") {
    sh $env:BUILD_SOURCESDIRECTORY/build-python.sh install --user --mingw ; Check-Output $?
  } else {
    sh $env:BUILD_SOURCESDIRECTORY/build-python.sh install --user; Check-Output $?
  }
}

if (($env:TASK -eq "sdist") -or (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python"))) {
  # cannot test C API with "sdist" task
  $tests = $env:BUILD_SOURCESDIRECTORY + "/tests/python_package_test"
} else {
  $tests = $env:BUILD_SOURCESDIRECTORY + "/tests"
}
if ($env:TASK -eq "bdist") {
  # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
  $env:LIGHTGBM_TEST_DUAL_CPU_GPU = "1"
}

pytest $tests ; Check-Output $?

if (($env:TASK -eq "regular") -or (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python"))) {
  cd $env:BUILD_SOURCESDIRECTORY/examples/python-guide
  @("import matplotlib", "matplotlib.use('Agg')") + (Get-Content "plot_example.py") | Set-Content "plot_example.py"
  (Get-Content "plot_example.py").replace('graph.render(view=True)', 'graph.render(view=False)') | Set-Content "plot_example.py"  # prevent interactive window mode
  conda install -y -n $env:CONDA_ENV "h5py>=3.10" "ipywidgets>=8.1.2" "notebook>=7.1.2"
  foreach ($file in @(Get-ChildItem *.py)) {
    @("import sys, warnings", "warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))") + (Get-Content $file) | Set-Content $file
    python $file ; Check-Output $?
  }  # run all examples
  cd $env:BUILD_SOURCESDIRECTORY/examples/python-guide/notebooks
  (Get-Content "interactive_plot_example.ipynb").replace('INTERACTIVE = False', 'assert False, \"Interactive mode disabled\"') | Set-Content "interactive_plot_example.ipynb"
  jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb ; Check-Output $?  # run all notebooks
}
