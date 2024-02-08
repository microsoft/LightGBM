# coding: utf-8
"""Find the path to LightGBM dynamic library files."""
from pathlib import Path
from platform import system
from typing import List

__all__: List[str] = []


def find_lib_path() -> List[str]:
    """Find the path to LightGBM library files.

    Returns
    -------
    lib_path: list of str
       List of all found library paths to LightGBM.
    """
    curr_path = Path(__file__).absolute()
    dll_path = [curr_path.parents[1],
                curr_path.parents[0] / 'bin',
                curr_path.parents[0] / 'lib']
    if system() in ('Windows', 'Microsoft'):
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
