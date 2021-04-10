# coding: utf-8
"""Find the path to LightGBM dynamic library files."""
import os
from platform import system
from typing import List


def find_lib_path() -> List[str]:
    """Find the path to LightGBM library files.

    Returns
    -------
    lib_path: list of strings
       List of all found library paths to LightGBM.
    """
    if os.environ.get('LIGHTGBM_BUILD_DOC', False):
        # we don't need lib_lightgbm while building docs
        return []

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path,
                os.path.join(curr_path, '../../'),
                os.path.join(curr_path, 'compile'),
                os.path.join(curr_path, '../compile'),
                os.path.join(curr_path, '../../lib/')]
    if system() in ('Windows', 'Microsoft'):
        dll_path.append(os.path.join(curr_path, '../compile/Release/'))
        dll_path.append(os.path.join(curr_path, '../compile/windows/x64/DLL/'))
        dll_path.append(os.path.join(curr_path, '../../Release/'))
        dll_path.append(os.path.join(curr_path, '../../windows/x64/DLL/'))
        dll_path = [os.path.join(p, 'lib_lightgbm.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'lib_lightgbm.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_path:
        dll_path = [os.path.realpath(p) for p in dll_path]
        new_line = "\n"
        raise Exception(f'Cannot find lightgbm library file in following paths:{new_line}{new_line.join(dll_path)}')
    return lib_path
