#!/usr/bin/env python
# Pass a single commandline argument to this executable: the path to the
# 'version.py' file containing the version number under the variable 'version'.
"""Print the current version of the module."""
import os
import sys
import importlib.machinery
import importlib.util

if len(sys.argv) < 2:
    raise ValueError("No filepath provided.")
version_file_path = sys.argv[1]


# Get the version number
def _import_from_path(module_name: str, file_path: str):  # noqa: ANN202
    spec: importlib.machinery.ModuleSpec = (
        importlib.util.spec_from_file_location(module_name, file_path)
    )  # ty: ignore [invalid-assignment]
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # ty: ignore [unresolved-attribute]
    return module


version: str = _import_from_path(
    "version", version_file_path
).version

print(version)
