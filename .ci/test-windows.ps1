function Assert-Output {
    param( [Parameter(Mandatory = $true)][bool]$success )
    if (-not $success) {
        $host.SetShouldExit(-1)
        exit 1
    }
}

$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
pip install --upgrade pip

Set-Location "$env:BUILD_SOURCESDIRECTORY"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle
pip install -v lightgbm --no-binary lightgbm

$tests = "$env:BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests ; Assert-Output $?
