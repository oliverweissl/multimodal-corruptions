#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=methodology python -c "
from data_selector import DataSelector
DataSelector().run_selection()
"