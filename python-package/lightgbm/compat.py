# coding: utf-8
"""Compatibility"""
from __future__ import absolute_import

import inspect
import sys

is_py3 = (sys.version_info[0] == 3)

if is_py3:
    string_type = str
    numeric_types = (int, float, bool)
    integer_types = int
    range_ = range
    argc_ = lambda func: len(inspect.signature(func).parameters)
else:
    string_type = basestring
    numeric_types = (int, long, float, bool)
    integer_types = (int, long)
    range_ = xrange
    argc_ = lambda func: len(inspect.getargspec(func).args)
