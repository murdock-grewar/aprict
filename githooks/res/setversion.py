#!/usr/bin/env python
# Pass a two commandline arguments to this executable
# $1 - the path to the 'version.py' file containing the version number under the variable 'version'.
# $2 - the new version string
"""Set the current version of the module."""
import sys

if len(sys.argv) < 2:
    raise ValueError("No filepath or version string provided.")
if len(sys.argv) < 3:
    raise ValueError("No version string provided.")
filepath = sys.argv[1]
new_version: str = sys.argv[2]

# Read the current version file
with open(filepath) as f:
    lines = f.readlines()

# Pair the line numbers with the line contents. (Make them all lists instead of tuples so I can mutate them.)
numlines = list(map(list, enumerate(lines)))

# Find candidate lines which might declare the version number.
candidate_lines = [numline for numline in numlines if numline[1].startswith('version')]

# If there isn't exactly one (1) candidate found, then abort.
num_cands = len(candidate_lines)
if num_cands < 1:
    raise ValueError("Could not find file line which declares the version. The line should begin with 'version'.")
elif num_cands > 1:
    raise ValueError("Found multiple file lines which may declare the version. Only one line should begin with 'version'.")

# Get the index of the line to update
idx_version_line = candidate_lines[0][0]

# Set the new version line
numlines[idx_version_line][1] = f"version = '{new_version:s}'"

# Write out the new file
with open(filepath, 'w') as f:
    f.writelines([numline[1] for numline in numlines])
    f.write('\n')
