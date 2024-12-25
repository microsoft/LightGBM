$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)

$env:MINICONDA="C:\Miniconda3"
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

bash.exe $env:BUILD_SOURCESDIRECTORY/.ci/test.sh
