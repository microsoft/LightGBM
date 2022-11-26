from setuptools import find_packages
from skbuild import setup

setup(
    cmake_languages=('C', 'CXX'),
    include_package_data=True,
    packages=find_packages(),
    # ld: can't write output file to '/Users/jlamb/repos/LightGBM/lightgbm-python/lightgbm'
    # because that path is a directory
    package_dir={"lightgbm": "lightgbm"},
    zip_saafe=False
)

# ld: can't write output file to '/Users/jlamb/repos/LightGBM/lightgbm-python/lightgbm'
# because that path is a directory
