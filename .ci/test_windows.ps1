function Check-Output {
  param( [bool]$success )
  if (!$success) {
    $host.SetShouldExit(-1)
    Exit -1
  }
}

if ($env:TASK -eq "regular") {
  mkdir $env:BUILD_SOURCESDIRECTORY/build; cd $env:BUILD_SOURCESDIRECTORY/build
  cmake -DCMAKE_GENERATOR_PLATFORM=x64 .. ; cmake --build . --target ALL_BUILD --config Release ; Check-Output $?
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
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py bdist_wheel --plat-name=win-amd64 --universal ; Check-Output $?
  cd dist; pip install @(Get-ChildItem *.whl) ; Check-Output $?
  cp @(Get-ChildItem *.whl) $env:BUILD_ARTIFACTSTAGINGDIRECTORY
}

$tests = $env:BUILD_SOURCESDIRECTORY + $(If ($env:TASK -eq "sdist") {"/tests/python_package_test"} Else {"/tests"})  # cannot test C API with "sdist" task
pytest $tests ; Check-Output $?

if ($env:TASK -eq "regular") {
  cd $env:BUILD_SOURCESDIRECTORY/examples/python-guide
  @("import matplotlib", "matplotlib.use('Agg')") + (Get-Content "plot_example.py") | Set-Content "plot_example.py"
  (Get-Content "plot_example.py").replace('graph.render(view=True)', 'graph.render(view=False)') | Set-Content "plot_example.py"
  foreach ($file in @(Get-ChildItem *.py)) {
    @("import sys, warnings", "warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))") + (Get-Content $file) | Set-Content $file
    python $file ; Check-Output $?
  }  # run all examples
  cd $env:BUILD_SOURCESDIRECTORY/examples/python-guide/notebooks
  conda install -y -n $env:CONDA_ENV ipywidgets notebook
  jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb ; Check-Output $?  # run all notebooks
}
