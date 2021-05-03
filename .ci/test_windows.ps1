function Check-Output {
  param( [bool]$success )
  if (!$success) {
    $host.SetShouldExit(-1)
    Exit -1
  }
}

# unify environment variable for Azure DevOps and AppVeyor
if (Test-Path env:APPVEYOR) {
  $env:APPVEYOR = "true"
}

if ($env:TASK -eq "r-package") {
  & $env:BUILD_SOURCESDIRECTORY\.ci\test_r_package_windows.ps1 ; Check-Output $?
  Exit 0
}

if ($env:TASK -eq "cpp-tests") {
  mkdir $env:BUILD_SOURCESDIRECTORY/build; cd $env:BUILD_SOURCESDIRECTORY/build
  cmake -DBUILD_CPP_TEST=ON -DUSE_OPENMP=OFF -A x64 ..
  cmake --build . --target testlightgbm --config Debug ; Check-Output $?
  Start-Process -FilePath "./../Debug/testlightgbm.exe" -NoNewWindow -Wait ; Check-Output $?
  Exit 0
}

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda
conda create -q -y -n $env:CONDA_ENV python=$env:PYTHON_VERSION ; Check-Output $?
if ($env:TASK -ne "bdist") {
  conda activate $env:CONDA_ENV
}

if ($env:TASK -eq "swig") {
  $env:JAVA_HOME = $env:JAVA_HOME_8_X64  # there is pre-installed Zulu OpenJDK-8 somewhere
  $ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
  Invoke-WebRequest -Uri "https://github.com/microsoft/LightGBM/releases/download/v2.0.12/swigwin-4.0.2.zip" -OutFile $env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip -UserAgent "NativeHost"
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory("$env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip", "$env:BUILD_SOURCESDIRECTORY/swig")
  $env:PATH += ";$env:BUILD_SOURCESDIRECTORY/swig/swigwin-4.0.2"
  mkdir $env:BUILD_SOURCESDIRECTORY/build; cd $env:BUILD_SOURCESDIRECTORY/build
  cmake -A x64 -DUSE_SWIG=ON .. ; cmake --build . --target ALL_BUILD --config Release ; Check-Output $?
  if ($env:AZURE -eq "true") {
    cp $env:BUILD_SOURCESDIRECTORY/build/lightgbmlib.jar $env:BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_win.jar
  }
  Exit 0
}

conda install -q -y -n $env:CONDA_ENV joblib matplotlib numpy pandas psutil pytest python-graphviz scikit-learn scipy ; Check-Output $?

if ($env:TASK -eq "regular") {
  mkdir $env:BUILD_SOURCESDIRECTORY/build; cd $env:BUILD_SOURCESDIRECTORY/build
  cmake -A x64 .. ; cmake --build . --target ALL_BUILD --config Release ; Check-Output $?
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py install --precompile ; Check-Output $?
  cp $env:BUILD_SOURCESDIRECTORY/Release/lib_lightgbm.dll $env:BUILD_ARTIFACTSTAGINGDIRECTORY
  cp $env:BUILD_SOURCESDIRECTORY/Release/lightgbm.exe $env:BUILD_ARTIFACTSTAGINGDIRECTORY
}
elseif ($env:TASK -eq "sdist") {
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py sdist --formats gztar ; Check-Output $?
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
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py bdist_wheel --integrated-opencl --plat-name=win-amd64 --python-tag py3 ; Check-Output $?
  cd dist; pip install --user @(Get-ChildItem *.whl) ; Check-Output $?
  cp @(Get-ChildItem *.whl) $env:BUILD_ARTIFACTSTAGINGDIRECTORY
} elseif (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python")) {
  cd $env:BUILD_SOURCESDIRECTORY\python-package
  if ($env:COMPILER -eq "MINGW") {
    python setup.py install --mingw ; Check-Output $?
  } else {
    python setup.py install ; Check-Output $?
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
  foreach ($file in @(Get-ChildItem *.py)) {
    @("import sys, warnings", "warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))") + (Get-Content $file) | Set-Content $file
    python $file ; Check-Output $?
  }  # run all examples
  cd $env:BUILD_SOURCESDIRECTORY/examples/python-guide/notebooks
  conda install -q -y -n $env:CONDA_ENV ipywidgets notebook
  jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb ; Check-Output $?  # run all notebooks
}
