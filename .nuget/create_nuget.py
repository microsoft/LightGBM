import os
import sys
import distutils
from distutils import file_util

if __name__ == "__main__":
    source = sys.argv[1]
    if not os.path.exists("runtimes/linux-x64/native"):
        os.makedirs("runtimes/linux-x64/native")
    if not os.path.exists("runtimes/osx-x64/native"):
        os.makedirs("runtimes/osx-x64/native")
    if not os.path.exists("runtimes/win-x64/native"):
        os.makedirs("runtimes/win-x64/native")
    distutils.file_util.copy_file(source+"/lib_lightgbm.so", "runtimes/linux-x64/native/lib_lightgbm.so")
    distutils.file_util.copy_file(source+"/lib_lightgbm.dylib", "runtimes/osx-x64/native/lib_lightgbm.dylib")
    distutils.file_util.copy_file(source+"/lib_lightgbm.dll", "runtimes/win-x64/native/lib_lightgbm.dll")
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
        <copyright>Copyright 2018 @ Microsoft</copyright>
        <tags>machine-learning data-mining distributed native boosting gbdt</tags>
        <dependencies> </dependencies>
    </metadata>
        <files>
        <file src="runtimes\**" target="runtimes"/>
        </files>
    </package>
    ''' % (version)
    nuget_file = open("LightGBM.nuspec", "w")
    nuget_file.write(nuget_str)
    nuget_file.close()
