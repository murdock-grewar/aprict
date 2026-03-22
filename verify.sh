#!/usr/bin/env bash
source .venv/bin/activate
SKIP='version-increment' pre-commit run --all-files
deactivate
