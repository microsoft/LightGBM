#!/usr/bin/bash
cd python-package
python setup.py bdist_wheel
pip uninstall -y lightgbm
cd dist
pip install lightgbm-*.whl
cd ../..
