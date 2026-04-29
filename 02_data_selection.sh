#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=methodology python -c "
from initialize.data_selector import DataSelector
DataSelector().run_selection()
"