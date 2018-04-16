import os
import sys
import distutils
from distutils import file_util

if not os.path.exists("lib"):
    os.makedirs("lib")
if not os.path.exists("lib/native"):
    os.makedirs("lib/native")
if not os.path.exists("lib/net40"):
    os.makedirs("lib/net40")
if not os.path.exists("lib/net45"):
    os.makedirs("lib/net45")
distutils.file_util.copy_file("../Release/lightgbm.exe", "./lib/")
distutils.file_util.copy_file("../Release/lib_lightgbm.dll", "./lib/")
version = open('../VERSION.txt').read().strip()
nuget_str = '''<?xml version="1.0"?>
<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
  <metadata>
    <id>LightGBM</id>
    <version>%s</version>
    <authors>Guolin Ke</authors>
    <owners>Guolin Ke</owners>
    <licenseUrl>https://github.com/Microsoft/LightGBM/blob/master/LICENSE</licenseUrl>
    <projectUrl>https://github.com/Microsoft/LightGBM</projectUrl>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <description>A fast, distributed, high performance gradient boosting framework</description>
    <copyright>Copyright 2017 @ Microsoft</copyright>
    <tags>machine-learning data-mining distributed native boosting gbdt</tags>
    <dependencies> </dependencies>
  </metadata>
      <files>
      <file src="lib\**" target="lib"/>
    </files>
</package>
''' % (version)
nuget_file = open("LightGBM.nuspec", "w")
nuget_file.write(nuget_str)
nuget_file.close()
