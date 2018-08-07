import os
import sys

from distutils.file_util import copy_file


if __name__ == "__main__":
    source = sys.argv[1]
    current_dir = os.path.abspath(os.path.dirname(__file__))
    linux_folder_path = os.path.join(current_dir,"lib", "netstandard2.0", "runtimes", "linux-x64", "native")
    if not os.path.exists(linux_folder_path):
        os.makedirs(linux_folder_path)
    osx_folder_path = os.path.join(current_dir, "lib", "netstandard2.0", "runtimes", "osx-x64", "native")
    if not os.path.exists(osx_folder_path):
        os.makedirs(osx_folder_path)
    windows_folder_path = os.path.join(current_dir, "lib", "netstandard2.0", "runtimes", "win-x64", "native")
    if not os.path.exists(windows_folder_path):
        os.makedirs(windows_folder_path)
    net46_folder_path = os.path.join(current_dir, "lib", "net46")
    if not os.path.exists(net46_folder_path):
        os.makedirs(net46_folder_path)
    copy_file(os.path.join(source, "lib_lightgbm.so"), os.path.join(linux_folder_path, "lib_lightgbm.so"))
    copy_file(os.path.join(source, "lib_lightgbm.dylib"), os.path.join(osx_folder_path, "lib_lightgbm.dylib"))
    copy_file(os.path.join(source, "lib_lightgbm.dll"), os.path.join(windows_folder_path, "lib_lightgbm.dll"))
    copy_file(os.path.join(source, "lib_lightgbm.dll"), os.path.join(net46_folder_path, "lib_lightgbm.dll"))
    copy_file(os.path.join(source, "lightgbm.exe"), os.path.join(windows_folder_path, "lightgbm.exe"))
    copy_file(os.path.join(source, "lightgbm.exe"), os.path.join(net46_folder_path, "lightgbm.exe"))
    version = open(os.path.join(current_dir, os.path.pardir, 'VERSION.txt')).read().strip()
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
        <file src="lib\**" target="lib"/>
        </files>
    </package>
    ''' % version
    with open(os.path.join(current_dir, "LightGBM.nuspec"), "w") as nuget_file:
        nuget_file.write(nuget_str)
