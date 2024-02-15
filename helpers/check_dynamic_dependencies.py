# coding: utf-8
"""Helper script for checking versions in the dynamic symbol table.

This script checks that LightGBM library is linked to the appropriate symbol versions.

Linking to newer symbol versions at compile time is problematic because it could result
in built artifacts being unusable on older platforms.

Version history for these symbols can be found at the following:

* GLIBC: https://sourceware.org/glibc/wiki/Glibc%20Timeline
* GLIBCXX: https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
* OMP/GOMP: https://github.com/gcc-mirror/gcc/blob/master/libgomp/libgomp.map
"""
import re
import sys
from pathlib import Path


def check_dependencies(objdump_string: str) -> None:
    """Check the dynamic symbol versions.

    Parameters
    ----------
    objdump_string : str
        The dynamic symbol table entries of the file (result of `objdump -T` command).
    """
    GLIBC_version = re.compile(r"0{16}[ \(\t]+GLIBC_(\d{1,2})[.](\d{1,3})[.]?\d{,3}[ \)\t]+")
    versions = GLIBC_version.findall(objdump_string)
    assert len(versions) > 1
    for major, minor in versions:
        error_msg = f"found unexpected GLIBC version: '{major}.{minor}'"
        assert int(major) <= 2, error_msg
        assert int(minor) <= 28, error_msg

    GLIBCXX_version = re.compile(r"0{16}[ \(\t]+GLIBCXX_(\d{1,2})[.](\d{1,2})[.]?(\d{,3})[ \)\t]+")
    versions = GLIBCXX_version.findall(objdump_string)
    assert len(versions) > 1
    for major, minor, patch in versions:
        error_msg = f"found unexpected GLIBCXX version: '{major}.{minor}.{patch}'"
        assert int(major) == 3, error_msg
        assert int(minor) == 4, error_msg
        assert patch == "" or int(patch) <= 22, error_msg

    GOMP_version = re.compile(r"0{16}[ \(\t]+G?OMP_(\d{1,2})[.](\d{1,2})[.]?\d{,3}[ \)\t]+")
    versions = GOMP_version.findall(objdump_string)
    assert len(versions) > 1
    for major, minor in versions:
        error_msg = f"found unexpected OMP/GOMP version: '{major}.{minor}'"
        assert int(major) <= 4, error_msg
        assert int(minor) <= 5, error_msg


if __name__ == "__main__":
    check_dependencies(Path(sys.argv[1]).read_text(encoding="utf-8"))
