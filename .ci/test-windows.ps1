function Assert-Output {
    param( [Parameter(Mandatory = $true)][bool]$success )
    if (-not $success) {
        $host.SetShouldExit(-1)
        exit 1
    }
}

$env:CONDA_ENV = "test-env"
$env:LGB_VER = (Get-Content $env:BUILD_SOURCESDIRECTORY\VERSION.txt).trim()
# Use custom temp directory to avoid
# > warning MSB8029: The Intermediate directory or Output directory cannot reside under the Temporary directory
# > as it could lead to issues with incremental build.
# And make sure this directory is always clean
$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)

if ($env:TASK -eq "r-package") {
    & .\.ci\test-r-package-windows.ps1 ; Assert-Output $?
    Exit 0
}

if ($env:TASK -eq "cpp-tests") {
    cmake -B build -S . -DBUILD_CPP_TEST=ON -DUSE_DEBUG=ON -A x64
    cmake --build build --target testlightgbm --config Debug ; Assert-Output $?
    .\Debug\testlightgbm.exe ; Assert-Output $?
    Exit 0
}

if ($env:TASK -eq "swig") {
    $env:JAVA_HOME = $env:JAVA_HOME_8_X64  # there is pre-installed Eclipse Temurin 8 somewhere
    $ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
    $params = @{
        Uri = "https://sourceforge.net/projects/swig/files/latest/download"
        OutFile = "$env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip"
        UserAgent = "curl"
    }
    Invoke-WebRequest @params
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        "$env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip",
        "$env:BUILD_SOURCESDIRECTORY/swig"
    ) ; Assert-Output $?
    $SwigFolder = Get-ChildItem -Directory -Name -Path "$env:BUILD_SOURCESDIRECTORY/swig"
    $env:PATH = @("$env:BUILD_SOURCESDIRECTORY/swig/$SwigFolder", "$env:PATH") -join ";"
    $BuildLogFileName = "$env:BUILD_SOURCESDIRECTORY\cmake_build.log"
    cmake -B build -S . -A x64 -DUSE_SWIG=ON *> "$BuildLogFileName" ; $build_succeeded = $?
    Write-Output "CMake build logs:"
    Get-Content -Path "$BuildLogFileName"
    Assert-Output $build_succeeded
    $checks = Select-String -Path "${BuildLogFileName}" -Pattern "-- Found SWIG.*${SwigFolder}/swig.exe"
    $checks_cnt = $checks.Matches.length
    if ($checks_cnt -eq 0) {
        Write-Output "Wrong SWIG version was found (expected '${SwigFolder}'). Check the build logs."
        Assert-Output $False
    }
    cmake --build build --target ALL_BUILD --config Release ; Assert-Output $?
    if ($env:AZURE -eq "true") {
        cp ./build/lightgbmlib.jar $env:BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_win.jar ; Assert-Output $?
    }
    Exit 0
}

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda "python=$env:PYTHON_VERSION[build=*_cp*]"

if ($env:PYTHON_VERSION -eq "3.7") {
    $env:CONDA_REQUIREMENT_FILE = "$env:BUILD_SOURCESDIRECTORY/.ci/conda-envs/ci-core-py37.txt"
} elseif ($env:PYTHON_VERSION -eq "3.8") {
    $env:CONDA_REQUIREMENT_FILE = "$env:BUILD_SOURCESDIRECTORY/.ci/conda-envs/ci-core-py38.txt"
} else {
    $env:CONDA_REQUIREMENT_FILE = "$env:BUILD_SOURCESDIRECTORY/.ci/conda-envs/ci-core.txt"
}

$condaParams = @(
    "-y",
    "-n", "$env:CONDA_ENV",
    "--file", "$env:CONDA_REQUIREMENT_FILE",
    "python=$env:PYTHON_VERSION[build=*_cp*]"
)
conda create @condaParams ; Assert-Output $?

if ($env:TASK -ne "bdist") {
    conda activate $env:CONDA_ENV
}

Set-Location "$env:BUILD_SOURCESDIRECTORY"
if ($env:TASK -eq "regular") {
    cmake -B build -S . -A x64 ; Assert-Output $?
    cmake --build build --target ALL_BUILD --config Release ; Assert-Output $?
    sh ./build-python.sh install --precompile ; Assert-Output $?
    cp ./Release/lib_lightgbm.dll "$env:BUILD_ARTIFACTSTAGINGDIRECTORY"
    cp ./Release/lightgbm.exe "$env:BUILD_ARTIFACTSTAGINGDIRECTORY"
} elseif ($env:TASK -eq "sdist") {
    sh ./build-python.sh sdist ; Assert-Output $?
    sh ./.ci/check-python-dists.sh ./dist ; Assert-Output $?
    Set-Location dist; pip install @(Get-ChildItem *.gz) -v ; Assert-Output $?
} elseif ($env:TASK -eq "bdist") {
    # Import the Chocolatey profile module so that the RefreshEnv command
    # invoked below properly updates the current PowerShell session environment.
    $module = "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1"
    Import-Module "$module" ; Assert-Output $?
    RefreshEnv

    Write-Output "Current OpenCL drivers:"
    Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors

    conda activate $env:CONDA_ENV
    sh "build-python.sh" bdist_wheel --integrated-opencl ; Assert-Output $?
    sh ./.ci/check-python-dists.sh ./dist ; Assert-Output $?
    Set-Location dist; pip install @(Get-ChildItem *py3-none-win_amd64.whl) ; Assert-Output $?
    cp @(Get-ChildItem *py3-none-win_amd64.whl) "$env:BUILD_ARTIFACTSTAGINGDIRECTORY"
} elseif (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python")) {
    if ($env:COMPILER -eq "MINGW") {
        sh ./build-python.sh install --mingw ; Assert-Output $?
    } else {
        sh ./build-python.sh install; Assert-Output $?
    }
}

if (($env:TASK -eq "sdist") -or (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python"))) {
    # cannot test C API with "sdist" task
    $tests = "$env:BUILD_SOURCESDIRECTORY/tests/python_package_test"
} else {
    $tests = "$env:BUILD_SOURCESDIRECTORY/tests"
}
if ($env:TASK -eq "bdist") {
    # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
    $env:LIGHTGBM_TEST_DUAL_CPU_GPU = "1"
}

pytest $tests ; Assert-Output $?

if (($env:TASK -eq "regular") -or (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python"))) {
    Set-Location "$env:BUILD_SOURCESDIRECTORY/examples/python-guide"
    @("import matplotlib", "matplotlib.use('Agg')") + (Get-Content "plot_example.py") | Set-Content "plot_example.py"
    # Prevent interactive window mode
    (Get-Content "plot_example.py").replace(
        'graph.render(view=True)',
        'graph.render(view=False)'
    ) | Set-Content "plot_example.py"
    conda install -y -n $env:CONDA_ENV "h5py>=3.10" "ipywidgets>=8.1.2" "notebook>=7.1.2"
    # Run all examples
    foreach ($file in @(Get-ChildItem *.py)) {
        @(
            "import sys, warnings",
            -join @(
                "warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: ",
                "sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))"
            )
        ) + (Get-Content $file) | Set-Content $file
        python $file ; Assert-Output $?
    }
    # Run all notebooks
    Set-Location "$env:BUILD_SOURCESDIRECTORY/examples/python-guide/notebooks"
    (Get-Content "interactive_plot_example.ipynb").replace(
        'INTERACTIVE = False',
        'assert False, \"Interactive mode disabled\"'
    ) | Set-Content "interactive_plot_example.ipynb"
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb ; Assert-Output $?
}
