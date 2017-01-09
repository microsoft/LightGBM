# coding: utf-8
"""Compatibility"""

from __future__ import absolute_import

import sys

is_py3 = (sys.version_info[0] == 3)

if is_py3:
    string_type = str
    numeric_types = (int, float, bool)
    integer_types = int
else:
    string_type = basestring
    numeric_types = (int, long, float, bool)
    integer_types = (int, long)
