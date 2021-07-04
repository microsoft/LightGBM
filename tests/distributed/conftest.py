import os

TESTS_DIR = os.path.dirname(__file__)
default_exec_file = os.path.abspath(os.path.join(TESTS_DIR, '..', '..', 'lightgbm'))


def pytest_addoption(parser):
    parser.addoption('--execfile', action='store', default=default_exec_file)
