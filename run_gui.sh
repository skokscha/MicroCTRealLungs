#!/bin/bash
# GUI RVE500 для Linux/WSL:  bash run_gui.sh
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python rve_gui.py
else
  python3 rve_gui.py
fi
