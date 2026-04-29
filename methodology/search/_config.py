QWEN_SCALE_FACTOR = 1000

DATASET_PATH = "dataset/2017/ILSVRC/Data/DET/val"
ANNOTATIONS_PATH = "dataset/2017/ILSVRC/Annotations/DET/val"
MAT_FILE_PATH = "dataset/2017/ILSVRC/devkit/data/meta_det.mat"

RESULTS_DIR = "results/selection"
OUTPUT_BASE_DIRS = {
    "multi": "multimodal/",
    "image": "unimodal/image/",
    "text":  "unimodal/text/",
}

BASELINE_FAIL_FILENAME = "baseline_fail.json"
PARETO_FILENAME = "pareto_front.json"
BEST_RESULT_FILENAME = "best_result.json"

# Sequential application order matters — see inline comments
IMAGE_ATTACKS = [
    "elastic",          # Structural deformation on clean pixels first
    "gaussian_noise",   # Noise on clean/deformed pixels
    "defocus_blur",     # Absorbs noise, degrades spatial features
    "motion_blur",      # Compounds blur, more natural on soft input
    "fog_filter",       # Global overlay, physically plausible after blur
    "snow_filter",      # Overlay, after blur (requires utils/OverlayImages/snow.png)
    "contrast",         # Contrast shift after spatial corruptions
    "false_color",      # Color channel manipulation
    "grayscale",        # Color reduction
    "cutout",           # Occlusion patches (bboxes passed from evaluator)
    "pixelate",         # Locks in all prior damage at reduced resolution
    "jpeg_filter",      # ALWAYS LAST, simulates real compression
]

TEXT_ATTACKS = [
    "homophone",        # Dictionary lookup, needs clean tokens
    "synonym",          # Dictionary lookup, needs clean tokens
    "ata_saliency",     # Saliency-targeted typos, needs word boundaries
    "fragmentation",    # Splits words, after all word-level ops
    "character_noise",  # ALWAYS LAST, character-level, breaks everything upstream
]

N_IMG = len(IMAGE_ATTACKS)
N_TXT = len(TEXT_ATTACKS)
N_VAR = N_IMG + N_TXT
N_OBJ = 3
