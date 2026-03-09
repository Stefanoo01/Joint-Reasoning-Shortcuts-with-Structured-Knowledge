#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/third_party/rsbench-code:${PYTHONPATH:-}"
echo "PYTHONPATH updated with rsbench-code"
python -c "import sys; print('rsbench-code in path:', any('rsbench-code' in p for p in sys.path))"