# coding: utf-8
"""Helper script for checking versions in the dynamic symbol table.

This script checks that LightGBM library is linked to the appropriate symbol versions.
"""
import re
import sys
from pathlib import Path


def check_dependicies(objdump_string: str) -> None:
    """Check the dynamic symbol versions.

    Parameters
    ----------
    objdump_string : str
        The dynamic symbol table entries of the file (result of `objdump -T` command).
    """
    GLIBC_version = re.compile(r'0{16}[ \t]+GLIBC_(\d{1,2})[.](\d{1,3})[.]?\d{,3}[ \t]+')
    versions = GLIBC_version.findall(objdump_string)
    assert len(versions) > 1
    for major, minor in versions:
        assert int(major) <= 2
        assert int(minor) <= 14

    GLIBCXX_version = re.compile(r'0{16}[ \t]+GLIBCXX_(\d{1,2})[.](\d{1,2})[.]?(\d{,3})[ \t]+')
    versions = GLIBCXX_version.findall(objdump_string)
    assert len(versions) > 1
    for major, minor, patch in versions:
        assert int(major) == 3
        assert int(minor) == 4
        assert patch == '' or int(patch) <= 19

    GOMP_version = re.compile(r'0{16}[ \t]+G?OMP_(\d{1,2})[.](\d{1,2})[.]?\d{,3}[ \t]+')
    versions = GOMP_version.findall(objdump_string)
    assert len(versions) > 1
    for major, minor in versions:
        assert int(major) == 1
        assert int(minor) == 0


if __name__ == "__main__":
    check_dependicies(Path(sys.argv[1]).read_text(encoding='utf-8'))
