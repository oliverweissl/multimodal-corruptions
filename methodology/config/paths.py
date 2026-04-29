DATASET_PATH = "dataset/2017/ILSVRC/Data/DET/val"
ANNOTATIONS_PATH = "dataset/2017/ILSVRC/Annotations/DET/val"
MAT_FILE_PATH = "dataset/2017/ILSVRC/devkit/data/meta_det.mat"

RESULTS_DIR = "results/selection"

OUTPUT_BASE_DIRS = {
    "multi": "multimodal",
    "image": "unimodal/image",
    "text": "unimodal/text",
}

PARETO_FILE = "pareto_front.json"
BEST_FILE = "best_result.json"
BEST_IMAGE_FILE = "best_result.png"
BASELINE_FAIL_FILE = "baseline_fail.json"
