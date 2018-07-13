import os
import sys

from distutils.file_util import copy_file


if __name__ == "__main__":
    source = sys.argv[1]
    current_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(current_dir, "runtimes/linux-x64/native")):
        os.makedirs(os.path.join(current_dir, "runtimes/linux-x64/native"))
    if not os.path.exists(os.path.join(current_dir, "runtimes/osx-x64/native")):
        os.makedirs(os.path.join(current_dir, "runtimes/osx-x64/native"))
    if not os.path.exists(os.path.join(current_dir, "runtimes/win-x64/native")):
        os.makedirs(os.path.join(current_dir, "runtimes/win-x64/native"))
    copy_file(os.path.join(source, "lib_lightgbm.so"), os.path.join(current_dir, "runtimes/linux-x64/native/lib_lightgbm.so"))
    copy_file(os.path.join(source, "lib_lightgbm.dylib"), os.path.join(current_dir, "runtimes/osx-x64/native/lib_lightgbm.dylib"))
    copy_file(os.path.join(source, "lib_lightgbm.dll"), os.path.join(current_dir, "runtimes/win-x64/native/lib_lightgbm.dll"))
    version = open(os.path.join(current_dir, '../VERSION.txt')).read().strip()
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
    with open(os.path.join(current_dir, "LightGBM.nuspec"), "w") as nuget_file:
        nuget_file.write(nuget_str)
