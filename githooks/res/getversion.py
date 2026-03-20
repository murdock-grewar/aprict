#!/usr/bin/env python
# Pass a single commandline argument to this executable: the path to the 'version.py' file containing the version number under the varibale 'version'.
import sys
if len(sys.argv) < 2:
    raise ValueError("No filepath provided.")
filepath = sys.argv[1]
with open(filepath, 'r') as f:
    d = dict()
    exec(f.read(), d)
    version = d['version']
print(version)
