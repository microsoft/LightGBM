$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)


echo 44

$env:MINICONDA="C:\Miniconda3-x64"
$env:PATH="$env:MINICONDA;$env:MINICONDA\Scripts;$env:PATH"
$env:BUILD_SOURCESDIRECTORY="$env:APPVEYOR_BUILD_FOLDER"

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
conda config --remove channels defaults
conda config --add channels nodefaults
conda config --add channels conda-forge
conda config --set channel_priority strict
conda init powershell

python.exe -m pip install --upgrade pip

Set-Location "$env:BUILD_SOURCESDIRECTORY"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle

New-Item -Path "$env:BUILD_SOURCESDIRECTORY/.ci" -ItemType Directory
curl --output $env:BUILD_SOURCESDIRECTORY/.ci/setup.sh "https://raw.githubusercontent.com/microsoft/LightGBM/refs/heads/ci/test/.ci/setup.sh"


$tests = "$env:BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
