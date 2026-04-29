import os
import json
import re

# Disclaimer: This script was constructed with assistance from Gemini 3.1 Pro
# for code completion anbd debugging.

RESULTS_BASE_DIR = "results/selection"
OUTPUT_FILE = "label_mapping.json"

CATEGORIES = [
    os.path.join("single", "solo"),
    os.path.join("single", "multi"),
    "multi",
]


def extract_base_label(key):
    """Strip trailing '_N' suffixes used for duplicate instances."""
    return re.sub(r'_\d+$', '', key)


def collect_labels():
    labels = set()

    if not os.path.isdir(RESULTS_BASE_DIR):
        print(f"Warning: {RESULTS_BASE_DIR} not found. Run run.py first to generate selection data.")
        return labels

    # Scan all VLM subfolders (results/selection/<vlm>/)
    vlm_dirs = [
        os.path.join(RESULTS_BASE_DIR, d)
        for d in os.listdir(RESULTS_BASE_DIR)
        if os.path.isdir(os.path.join(RESULTS_BASE_DIR, d))
    ]

    if not vlm_dirs:
        print(f"Warning: no VLM subfolders found in {RESULTS_BASE_DIR}.")
        return labels

    for vlm_dir in vlm_dirs:
        for category in CATEGORIES:
            category_dir = os.path.join(vlm_dir, category)
            if not os.path.isdir(category_dir):
                continue

            for folder_name in sorted(os.listdir(category_dir)):
                json_path = os.path.join(category_dir, folder_name, "original.json")
                if not os.path.isfile(json_path):
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                gt = data.get("ground_truth", {})
                for key in gt:
                    labels.add(extract_base_label(key))

    return labels


def main():
    labels = collect_labels()
    print(f"Found {len(labels)} unique class labels.")

    mapping = {label: ["empty"] for label in sorted(labels)}

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Saved mapping to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
