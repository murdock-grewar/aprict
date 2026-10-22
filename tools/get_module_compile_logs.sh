#!/usr/bin/env bash
# Run this with the python environment activated.
export CUPY_CACHE_DIR="./tmp_cache_dir_for_testing"
if [ -d "$CUPY_CACHE_DIR" ]; then
    echo "Directory already exists: '$CUPY_CACHE_DIR'. Aborting." >&2
    exit 2
fi
mkdir "$CUPY_CACHE_DIR"
# Pipe to 'strings' to remove non-text characters, so grep processes it normally.
# Could also apply grep --text to the output, but I'll forget to  do this.
python -c 'import aprict, io, sys; aprict.cb.flatpanel.force_compile(log_stream=sys.stdout)' | strings
rm -r "$CUPY_CACHE_DIR"
