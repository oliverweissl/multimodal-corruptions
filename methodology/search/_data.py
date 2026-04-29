import os
import json
import logging
import numpy as np
from PIL import Image

from . import _config as _cfg
from .utils._image import resize_image_smart
from .utils._genome import decode_genome
from ._problem import AdversarialProblem

logger = logging.getLogger(__name__)


def load_sample(folder_path, max_resolution=1024):
    input_json = os.path.join(folder_path, "original.json")
    input_img = os.path.join(folder_path, "data_point.JPEG")
    if not os.path.exists(input_json) or not os.path.exists(input_img):
        raise FileNotFoundError(
            f"Missing original.json or data_point.JPEG in {folder_path}"
        )
    with open(input_json, "r") as f:
        base_data = json.load(f)
    raw_img = Image.open(input_img).convert("RGB")
    orig_w, orig_h = raw_img.size
    resized_img = resize_image_smart(raw_img, max_resolution)
    return {
        "clean_image_pil": resized_img,
        "original_prompt": base_data["prompt"],
        "gt_bboxes": base_data.get("ground_truth", {}),
        "filename": base_data.get("image", ""),
        "baseline_iou": float(base_data.get("IoU", 0.0)),
        "orig_dims": (orig_w, orig_h),
        "curr_dims": resized_img.size,
        "folder_path": folder_path,
    }


def get_all_sample_folders(results_dir=None):
    """
    Collect every valid data folder under:
      results_dir/single/solo/NNN
      results_dir/single/multi/NNN
      results_dir/multi/NNN
    Returns list of (folder_path, category_rel, folder_id).
    """
    if results_dir is None:
        results_dir = _cfg.RESULTS_DIR
    sample_folders = []
    categories = [
        os.path.join("single", "solo"),
        os.path.join("single", "multi"),
        "multi",
    ]
    for cat_rel in categories:
        cat_abs = os.path.join(results_dir, cat_rel)
        if not os.path.isdir(cat_abs):
            continue
        for fn in os.listdir(cat_abs):
            fp = os.path.join(cat_abs, fn)
            if not os.path.isdir(fp) or not fn.isdigit():
                continue
            if not os.path.exists(os.path.join(fp, "original.json")):
                continue
            if not os.path.exists(os.path.join(fp, "data_point.JPEG")):
                continue
            sample_folders.append((fp, cat_rel, fn))
    sample_folders.sort(key=lambda t: (t[1], int(t[2])))
    return sample_folders


def get_output_dir(category, folder_id, mode="multi"):
    return os.path.join(_cfg.OUTPUT_BASE_DIRS[mode], category, folder_id)


def is_already_processed(category, folder_id, mode="multi"):
    out = get_output_dir(category, folder_id, mode)
    return (
        os.path.exists(os.path.join(out, _cfg.BEST_RESULT_FILENAME))
        or os.path.exists(os.path.join(out, _cfg.BASELINE_FAIL_FILENAME))
    )


def save_baseline_fail(category, folder_id, mode, baseline_iou, sample_data):
    out = get_output_dir(category, folder_id, mode)
    os.makedirs(out, exist_ok=True)
    record = {
        "status": "baseline_fail",
        "baseline_iou": float(f"{baseline_iou:.5f}"),
        "data_source": {
            "folder_path": sample_data["folder_path"],
            "folder_id": folder_id,
            "category": category,
            "filename": sample_data["filename"],
        },
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
    }
    with open(os.path.join(out, _cfg.BASELINE_FAIL_FILENAME), "w") as f:
        json.dump(record, f, indent=4)


def _retrieve_cached_metrics(problem, x):
    key = AdversarialProblem._cache_key(x)
    cached = problem.metrics_cache.get(key)
    if cached is not None:
        return cached
    logger.warning("Cache miss: re-evaluating genome (single pass).")
    return problem.evaluator.evaluate_single(x, problem.sample_data)


def _build_best_meta(
    pareto_idx, sample_data, genome, F_vec, cached, iou_0,
    early_stopped, early_stop_gen, problem,
    folder_path, folder_id, category, runtime
):
    iou_adv = float(F_vec[0])
    img_dist = float(F_vec[1])
    txt_dist = float(F_vec[2])
    txt_sim = 1.0 - txt_dist

    return {
        "data_source": {
            "folder_path": folder_path,
            "folder_id": folder_id,
            "category": category,
            "filename": sample_data["filename"],
        },
        "pareto_index": pareto_idx,
        "budget_max": problem.budget_max,
        "batch_size": problem.batch_size,
        "early_stopped": early_stopped,
        "early_stop_generation": early_stop_gen,
        "total_evaluations": problem._eval_count,
        "skipped_evaluations": problem._skipped_count,
        "genome": genome,
        "runtime": runtime,
        "objectives": {
            "iou": float(f"{iou_adv:.5f}"),
            "img_dist": float(f"{img_dist:.5f}"),
            "txt_dist": float(f"{txt_dist:.5f}"),
            "txt_sim": float(f"{txt_sim:.5f}"),
        },
        "baseline_iou": float(f"{iou_0:.5f}"),
        "l2_distance": float(f"{np.linalg.norm(F_vec):.5f}"),
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
        "vlm_output": {
            "adversarial_prompt": cached["corrupt_prompt"],
            "raw_response": cached["vlm_response"],
            "parsed_predictions": cached["vlm_parsed"],
            "token_count": cached["token_count"],
            "raw_token_count": cached["raw_token_count"],
            "runtime_seconds": cached["runtime_seconds"],
        },
        "applied_img_corruptions": cached["applied_img_corruptions"],
        "applied_txt_corruptions": cached["applied_txt_corruptions"],
    }


def save_all_meta(
    result, sample_data, problem, output_dir, runtime,
    early_stopped=False, early_stop_gen=None,
):
    os.makedirs(output_dir, exist_ok=True)

    pareto_X = result.X
    pareto_F = result.F
    if pareto_X.ndim == 1:
        pareto_X = pareto_X.reshape(1, -1)
        pareto_F = pareto_F.reshape(1, -1)

    folder_path = sample_data["folder_path"]
    folder_id = sample_data["folder_id"]
    category = sample_data["category"]

    iou_0 = sample_data["baseline_iou"]

    pareto_records = []
    for i in range(len(pareto_X)):
        genome = decode_genome(pareto_X[i], problem.budget_max, mode=problem.mode)
        cached = _retrieve_cached_metrics(problem, pareto_X[i])

        iou_adv = float(pareto_F[i, 0])
        img_dist = float(pareto_F[i, 1])
        txt_dist = float(pareto_F[i, 2])
        txt_sim = 1.0 - txt_dist
        l2_dist = float(np.linalg.norm(pareto_F[i]))

        record = {
            "index": i,
            "genome": genome,
            "objectives": {
                "iou": float(f"{iou_adv:.5f}"),
                "img_dist": float(f"{img_dist:.5f}"),
                "txt_dist": float(f"{txt_dist:.5f}"),
                "txt_sim": float(f"{txt_sim:.5f}"),
            },
            "baseline_iou": float(f"{iou_0:.5f}"),
            "l2_distance": float(f"{l2_dist:.5f}"),
            "vlm_output": {
                "corrupt_prompt": cached["corrupt_prompt"],
                "raw_response": cached["vlm_response"],
                "parsed_predictions": cached["vlm_parsed"],
                "token_count": cached["token_count"],
                "raw_token_count": cached["raw_token_count"],
                "runtime_seconds": cached["runtime_seconds"],
            },
            "applied_img_corruptions": cached["applied_img_corruptions"],
            "applied_txt_corruptions": cached["applied_txt_corruptions"],
        }
        pareto_records.append(record)

    pareto_output = {
        "data_source": {
            "folder_path": folder_path,
            "folder_id": folder_id,
            "category": category,
            "filename": sample_data["filename"],
        },
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
        "baseline_iou": float(f"{iou_0:.5f}"),
        "budget_max": problem.budget_max,
        "batch_size": problem.batch_size,
        "early_stopped": early_stopped,
        "early_stop_generation": early_stop_gen,
        "total_evaluations": problem._eval_count,
        "skipped_evaluations": problem._skipped_count,
        "n_solutions": len(pareto_records),
        "solutions": pareto_records,
    }

    front_path = os.path.join(output_dir, _cfg.PARETO_FILENAME)
    with open(front_path, "w") as f:
        json.dump(pareto_output, f, indent=4)
    logger.info("Pareto front (%d solutions) -> %s", len(pareto_records), front_path)

    distances = np.linalg.norm(pareto_F, axis=1)
    best_idx = int(np.argmin(distances))

    best_x = pareto_X[best_idx]
    best_f = pareto_F[best_idx]
    best_genome = decode_genome(best_x, problem.budget_max, mode=problem.mode)
    best_cached = _retrieve_cached_metrics(problem, best_x)

    meta = _build_best_meta(
        pareto_idx=best_idx, sample_data=sample_data,
        genome=best_genome, F_vec=best_f, cached=best_cached, iou_0=iou_0,
        early_stopped=early_stopped, early_stop_gen=early_stop_gen,
        problem=problem, folder_path=folder_path, folder_id=folder_id,
        category=category, runtime=runtime
    )
    with open(os.path.join(output_dir, _cfg.BEST_RESULT_FILENAME), "w") as f:
        json.dump(meta, f, indent=4)