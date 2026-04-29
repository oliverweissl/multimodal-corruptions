#!/bin/bash
# Optional step between 02 and 03.
# Requires Ollama running locally with the target model pulled.
# Produces methodology/auxiliary_files/homophone_mapping.json and synonym_mapping.json.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=methodology python -m initialize.build_label_mapping
PYTHONPATH=methodology python -m initialize.generate_mappings "$@"
