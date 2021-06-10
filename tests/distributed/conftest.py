from pathlib import Path

default_exec_file = str(Path(__file__).absolute().parents[2] / 'lightgbm')


def pytest_addoption(parser):
    parser.addoption('--execfile', action='store', default=default_exec_file)
