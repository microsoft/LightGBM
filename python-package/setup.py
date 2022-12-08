from setuptools import find_packages
from skbuild import setup

setup(
    cmake_languages=('C', 'CXX'),
    include_package_data=True,
    packages=find_packages(),
    package_data={
        "lightgbm": [
            "LICENSE",
            "VERSION.txt"
        ]
    },
    # ld: can't write output file to '/Users/jlamb/repos/LightGBM/lightgbm-python/lightgbm'
    # because that path is a directory
    zip_safe=False
)
