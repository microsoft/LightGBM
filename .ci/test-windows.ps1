function Assert-Output {
    param( [Parameter(Mandatory = $true)][bool]$success )
    if (-not $success) {
        $host.SetShouldExit(-1)
        exit 1
    }
}

$env:CONDA_ENV = "test-env"
$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda "python=$env:PYTHON_VERSION[build=*cpython]"

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
    "python=$env:PYTHON_VERSION[build=*cpython]"
)
conda create @condaParams ; Assert-Output $?

if ($env:TASK -ne "bdist") {
    conda activate $env:CONDA_ENV
}

Set-Location "$env:BUILD_SOURCESDIRECTORY"
pip install -v lightgbm

$tests = "$env:BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests ; Assert-Output $?
