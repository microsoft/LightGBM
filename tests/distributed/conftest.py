from pathlib import Path

default_exec_dir = str(Path(__file__).absolute().parents[2])


def pytest_addoption(parser):
    parser.addoption('--execdir', action='store', default=default_exec_dir)
