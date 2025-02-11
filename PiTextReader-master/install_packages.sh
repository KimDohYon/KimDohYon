#!/bin/bash
# /deps에 있는 패키지들을 설치하는 스크립트

if [ -d "/workspace/finn/deps" ]; then
    echo "Installing packages from /workspace/finn/deps"
    pip install -e /workspace/finn/deps/qonnx
    pip install -e /workspace/finn/deps/finn-experimental
    pip install -e /workspace/finn/deps/brevitas
    pip install -e /workspace/finn/deps/pyverilator
else
    echo "Error: /workspace/finn/deps directory not found."
    exit 1
fi
