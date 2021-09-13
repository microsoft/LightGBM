# coding: utf-8
"""Find the path to LightGBM dynamic library files."""
from os import environ
from pathlib import Path
from platform import system
from typing import List


def find_lib_path() -> List[str]:
    """Find the path to LightGBM library files.

    Returns
    -------
    lib_path: list of str
       List of all found library paths to LightGBM.
    """
    if environ.get('LIGHTGBM_BUILD_DOC', False):
        # we don't need lib_lightgbm while building docs
        return []

    curr_path = Path(__file__).absolute().parent
    dll_path = [curr_path,
                curr_path.parents[1],
                curr_path / 'compile',
                curr_path.parent / 'compile',
                curr_path.parents[1] / 'lib']
    if system() in ('Windows', 'Microsoft'):
        dll_path.append(curr_path.parent / 'compile' / 'Release')
        dll_path.append(curr_path.parent / 'compile' / 'windows' / 'x64' / 'DLL')
        dll_path.append(curr_path.parents[1] / 'Release')
        dll_path.append(curr_path.parents[1] / 'windows' / 'x64' / 'DLL')
        dll_path = [p / 'lib_lightgbm.dll' for p in dll_path]
    else:
        dll_path = [p / 'lib_lightgbm.so' for p in dll_path]
    lib_path = [str(p) for p in dll_path if p.is_file()]
    if not lib_path:
        dll_path_joined = '\n'.join(map(str, dll_path))
        raise Exception(f'Cannot find lightgbm library file in following paths:\n{dll_path_joined}')
    return lib_path
