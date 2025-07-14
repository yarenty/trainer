#!/bin/bash
# install_wheels.sh
set +e
WHEEL_DIR="wheels"
for whl in "$WHEEL_DIR"/*.whl; do
    echo "Installing $whl"
    pip install --no-index --find-links="$WHEEL_DIR" "$whl"
    if [ $? -ne 0 ]; then
        echo "Failed to install $whl" >&2
    fi
done 