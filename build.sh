#!/usr/bin/env bash
bash githooks/uv-sync.sh  # ensure correct CUDA toolchain version is installed before building
bash githooks/flush-dtypes.sh
bash githooks/flush-ptx.sh
bash githooks/generate-cusrc.sh
bash githooks/generate-struct-ptx.sh
