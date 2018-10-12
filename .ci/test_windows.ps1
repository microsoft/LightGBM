function Check-Output {
  param( [int]$ExitCode )
  if ($ExitCode -ne 0) {
    $host.SetShouldExit($ExitCode)
    Exit -1
  }
}

if ($env:TASK -eq "regular") {
  mkdir $env:BUILD_SOURCESDIRECTORY/build; cd $env:BUILD_SOURCESDIRECTORY/build
  cmake -DCMAKE_GENERATOR_PLATFORM=x64 .. ; cmake --build . --target ALL_BUILD --config Release ; Check-Output $LastExitCode
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py install --precompile ; Check-Output $LastExitCode
  cp $env:BUILD_SOURCESDIRECTORY/Release/lib_lightgbm.dll $env:BUILD_ARTIFACTSTAGINGDIRECTORY
  cp $env:BUILD_SOURCESDIRECTORY/Release/lightgbm.exe $env:BUILD_ARTIFACTSTAGINGDIRECTORY
}
elseif ($env:TASK -eq "sdist") {
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py sdist --formats gztar ; Check-Output $LastExitCode
  cd dist; pip install @(Get-ChildItem *.gz) -v ; Check-Output $LastExitCode
}
elseif ($env:TASK -eq "bdist") {
  cd $env:BUILD_SOURCESDIRECTORY/python-package
  python setup.py bdist_wheel --plat-name=win-amd64 --universal ; Check-Output $LastExitCode
  cd dist; pip install @(Get-ChildItem *.whl) ; Check-Output $LastExitCode
  cp @(Get-ChildItem *.whl) $env:BUILD_ARTIFACTSTAGINGDIRECTORY
}

$tests = $env:BUILD_SOURCESDIRECTORY + $(If ($env:TASK -eq "sdist") {"/tests/python_package_test"} Else {"/tests"})  # cannot test C API with "sdist" task
pytest $tests ; Check-Output $LastExitCode

if ($env:TASK -eq "regular") {
  cd $env:BUILD_SOURCESDIRECTORY/examples/python-guide
  @("import matplotlib", "matplotlib.use('Agg')") + (Get-Content "plot_example.py") | Set-Content "plot_example.py"
  (Get-Content "plot_example.py").replace('graph.render(view=True)', 'graph.render(view=False)') | Set-Content "plot_example.py"
  foreach ($file in @(Get-ChildItem *.py)) {
    python $file ; Check-Output $LastExitCode
  }  # run all examples
}
