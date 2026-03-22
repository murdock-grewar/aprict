#!/usr/bin/env python
# Pass a single commandline argument to this executable: the path to the
# 'version.py' file containing the version number under the variable 'version'.
"""Print the current version of the module."""
import os
import sys
import importlib.machinery
import importlib.util
import runpy

if len(sys.argv) < 2:
    raise ValueError("No filepath provided.")
version_file_path = sys.argv[1]

version: str = runpy.run_path(version_file_path)['version']

print(version)
