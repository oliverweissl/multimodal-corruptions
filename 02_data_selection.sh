#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=methodology python -c "
from initialize.data_selector import DataSelector, DATASET_PATH, ANNOTATIONS_PATH, MAT_FILE_PATH, SEED, RESULTS_BASE_DIR
DataSelector(DATASET_PATH, ANNOTATIONS_PATH, MAT_FILE_PATH, SEED, RESULTS_BASE_DIR).run_selection()
"