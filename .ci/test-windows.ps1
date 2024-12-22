$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)

# setup for Python
conda init powershell
conda activate
conda config --set always_yes yes --set changeps1 no
python.exe -m pip install --upgrade pip

# export MINICONDA="C:\Miniconda3-x64" ; export PATH="$MINICONDA;$MINICONDA\bin;$PATH" ; export BUILD_SOURCESDIRECTORY="$APPVEYOR_BUILD_FOLDER" ; conda config --remove channels defaults ; conda config --add channels nodefaults ; conda config --add channels conda-forge ; conda config --set channel_priority strict ; conda init sh'

Set-Location "$env:BUILD_SOURCESDIRECTORY"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle
pip install lightgbm -v --no-binary lightgbm --config-settings=cmake.define.__BUILD_FOR_PYTHON=true --config-settings=cmake.args="-GVisual Studio 14 2015" --config-settings=cmake.args="-AWin32"


$tests = "$env:BUILD_SOURCESDIRECTORY/tests/python_package_test"

pytest $tests
