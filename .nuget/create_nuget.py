# coding: utf-8
"""Script for generating files with NuGet package metadata."""
import datetime
import os
import sys
from distutils.file_util import copy_file

if __name__ == "__main__":
    source = sys.argv[1]
    current_dir = os.path.abspath(os.path.dirname(__file__))
    linux_folder_path = os.path.join(current_dir, "runtimes", "linux-x64", "native")
    if not os.path.exists(linux_folder_path):
        os.makedirs(linux_folder_path)
    osx_folder_path = os.path.join(current_dir, "runtimes", "osx-x64", "native")
    if not os.path.exists(osx_folder_path):
        os.makedirs(osx_folder_path)
    windows_folder_path = os.path.join(current_dir, "runtimes", "win-x64", "native")
    if not os.path.exists(windows_folder_path):
        os.makedirs(windows_folder_path)
    build_folder_path = os.path.join(current_dir, "build")
    if not os.path.exists(build_folder_path):
        os.makedirs(build_folder_path)
    copy_file(os.path.join(source, "lib_lightgbm.so"), os.path.join(linux_folder_path, "lib_lightgbm.so"))
    copy_file(os.path.join(source, "lib_lightgbm.dylib"), os.path.join(osx_folder_path, "lib_lightgbm.dylib"))
    copy_file(os.path.join(source, "lib_lightgbm.dll"), os.path.join(windows_folder_path, "lib_lightgbm.dll"))
    copy_file(os.path.join(source, "lightgbm.exe"), os.path.join(windows_folder_path, "lightgbm.exe"))
    version = open(os.path.join(current_dir, os.path.pardir, 'VERSION.txt')).read().strip().replace('rc', '-rc')
    nuget_str = rf"""<?xml version="1.0"?>
    <package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
    <metadata>
        <id>LightGBM</id>
        <version>{version}</version>
        <authors>Guolin Ke</authors>
        <owners>Guolin Ke</owners>
        <licenseUrl>https://github.com/microsoft/LightGBM/blob/master/LICENSE</licenseUrl>
        <projectUrl>https://github.com/microsoft/LightGBM</projectUrl>
        <requireLicenseAcceptance>false</requireLicenseAcceptance>
        <description>A fast, distributed, high performance gradient boosting framework</description>
        <copyright>Copyright {datetime.datetime.now().year} @ Microsoft</copyright>
        <tags>machine-learning data-mining distributed native boosting gbdt</tags>
        <dependencies> </dependencies>
    </metadata>
        <files>
        <file src="build\**" target="build"/>
        <file src="runtimes\**" target="runtimes"/>
        </files>
    </package>
    """
    prop_str = r"""
    <Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
    <ItemGroup Condition="Exists('packages.config') OR
                            Exists('$(MSBuildProjectName).packages.config') OR
                            Exists('packages.$(MSBuildProjectName).config')">
        <Content Include="$(MSBuildThisFileDirectory)/../runtimes/win-x64/native/*.dll"
                Condition="'$(PlatformTarget)' == 'x64'">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
        <Visible>false</Visible>
        </Content>
        <Content Include="$(MSBuildThisFileDirectory)/../runtimes/win-x64/native/*.exe"
                Condition="'$(PlatformTarget)' == 'x64'">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
        <Visible>false</Visible>
        </Content>
    </ItemGroup>
    </Project>
    """
    target_str = r"""
    <Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
    <PropertyGroup>
        <EnableLightGBMUnsupportedPlatformTargetCheck Condition="'$(EnableLightGBMUnsupportedPlatformTargetCheck)' == ''">true</EnableLightGBMUnsupportedPlatformTargetCheck>
    </PropertyGroup>
    <Target Name="_LightGBMCheckForUnsupportedPlatformTarget"
            Condition="'$(EnableLightGBMUnsupportedPlatformTargetCheck)' == 'true'"
            AfterTargets="_CheckForInvalidConfigurationAndPlatform">
        <Error Condition="'$(PlatformTarget)' != 'x64' AND
                        ('$(OutputType)' == 'Exe' OR '$(OutputType)'=='WinExe') AND
                        !('$(TargetFrameworkIdentifier)' == '.NETCoreApp' AND '$(PlatformTarget)' == '')"
            Text="LightGBM currently supports 'x64' processor architectures. Please ensure your application is targeting 'x64'." />
    </Target>
    </Project>
    """
    with open(os.path.join(current_dir, "LightGBM.nuspec"), "w") as nuget_file:
        nuget_file.write(nuget_str)
    with open(os.path.join(current_dir, "build", "LightGBM.props"), "w") as prop_file:
        prop_file.write(prop_str)
    with open(os.path.join(current_dir, "build", "LightGBM.targets"), "w") as target_file:
        target_file.write(target_str)
