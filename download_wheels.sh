#!/bin/bash
# download_wheels.sh
set -e
REQ_FILE="requirements.txt"
WHEEL_DIR="wheels"
mkdir -p "$WHEEL_DIR"
pip download -r "$REQ_FILE" -d "$WHEEL_DIR" 