import json
import logging
import os

import numpy as np
from config.search import BASELINE_FAIL_FILENAME as _BASELINE_FAIL_FILE
from config.search import BEST_RESULT_FILENAME as _BEST_FILE
from config.search import BEST_RESULT_IMAGE_FILENAME as _BEST_IMAGE_FILE
from config.search import PARETO_FILENAME as _PARETO_FILE
from PIL import Image

from ._problem import PerturbationProblem
from .utils import decode_genome, ensure_rgb, resize_image_smart

logger = logging.getLogger(__name__)


def load_sample(folder_path: str, max_resolution: int = 1024) -> dict:
    """Load a sample folder into a runtime dict used by the evaluator.

    :param folder_path: Path to the sample directory containing ``original.json`` and ``data_point.JPEG``.
    :param max_resolution: Maximum image side length; the image is downscaled proportionally if exceeded.
    :returns: Dict with keys ``clean_image_pil``, ``original_prompt``, ``gt_bboxes``, ``orig_dims``, etc.
    :raises FileNotFoundError: If ``original.json`` or ``data_point.JPEG`` is missing.
    """
    input_json = os.path.join(folder_path, "original.json")
    input_img = os.path.join(folder_path, "data_point.JPEG")
    if not os.path.exists(input_json) or not os.path.exists(input_img):
        raise FileNotFoundError(f"Missing original.json or data_point.JPEG in {folder_path}")
    with open(input_json) as f:
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


def get_all_sample_folders(results_dir: str) -> list[tuple[str, str, str]]:
    """Discover all valid sample folders under the three annotation categories.

    :param results_dir: Root results directory produced by the data selector.
    :returns: Sorted list of ``(folder_path, category, folder_id)`` tuples.
    """
    sample_folders = []
    for cat_rel in (os.path.join("single", "solo"), os.path.join("single", "multi"), "multi"):
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


def get_output_dir(category: str, folder_id: str, output_base: str) -> str:
    """Build the output directory path for a given sample.

    :param category: Category relative path (e.g. ``"single/solo"``).
    :param folder_id: Numeric folder identifier string.
    :param output_base: Root output directory.
    :returns: Absolute-or-relative path for this sample's output.
    """
    return os.path.join(output_base, category, folder_id)


def is_already_processed(category: str, folder_id: str, output_base: str) -> bool:
    """Check whether a sample has already been processed by inspecting output files.

    :param category: Category relative path.
    :param folder_id: Numeric folder identifier string.
    :param output_base: Root output directory.
    :returns: ``True`` if a best-result or baseline-fail file exists for this sample.
    """
    out = get_output_dir(category, folder_id, output_base)
    return os.path.exists(os.path.join(out, _BEST_FILE)) or os.path.exists(
        os.path.join(out, _BASELINE_FAIL_FILE)
    )


def save_baseline_fail(output_dir: str, baseline_iou: float, sample_data: dict) -> None:
    """Write a baseline-fail record when the VLM cannot localise objects on the clean image.

    :param output_dir: Directory in which to write the record.
    :param baseline_iou: Clean-image IoU score that fell below the threshold.
    :param sample_data: Sample dict containing metadata and prompt.
    """
    os.makedirs(output_dir, exist_ok=True)
    record = {
        "status": "baseline_fail",
        "baseline_iou": float(f"{baseline_iou:.5f}"),
        "data_source": {
            "folder_path": sample_data["folder_path"],
            "folder_id": sample_data["folder_id"],
            "category": sample_data["category"],
            "filename": sample_data["filename"],
        },
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
        "predicted_bboxes": sample_data.get("baseline_preds", []),
    }
    with open(os.path.join(output_dir, _BASELINE_FAIL_FILE), "w") as f:
        json.dump(record, f, indent=4)


def _recreate_perturbed(x: np.ndarray, sample_data: dict, evaluator) -> Image.Image:
    """Recreate the perturbed image for a genome without re-running the VLM.

    :param x: Genome array (may be mode-specific length).
    :param sample_data: Sample dict with ``clean_image_pil`` and ``gt_bboxes``.
    :param evaluator: :class:`~search._evaluator.FitnessEvaluator` instance.
    :returns: PIL Image of the perturbed input.
    """
    full_x = evaluator._expand_genome(x)
    current_np = ensure_rgb(np.array(sample_data["clean_image_pil"])).copy()
    bboxes = [
        [v["xmin"], v["ymin"], v["xmax"], v["ymax"]] for v in sample_data["gt_bboxes"].values()
    ]
    for i, name in enumerate(evaluator.image_perturbations):
        kwargs = {"bboxes": bboxes} if name == "cutout" else {}
        current_np = evaluator.image_perturbator.apply_perturbation(
            current_np, name, scale=float(full_x[i]), **kwargs
        )
    return Image.fromarray(ensure_rgb(current_np).astype(np.uint8))


def _retrieve_cached_metrics(problem, x: np.ndarray) -> dict:
    """Fetch metrics from the problem cache, falling back to a fresh evaluation on miss.

    :param problem: :class:`~search._problem.PerturbationProblem` with a populated metrics cache.
    :param x: Genome array.
    :returns: Metrics dict (same format as :meth:`~search._evaluator.FitnessEvaluator.evaluate_single`).
    """
    cached = problem.metrics_cache.get(PerturbationProblem._cache_key(x))
    if cached is not None:
        return cached
    logger.warning("Cache miss: re-evaluating genome (single pass).")
    return problem.evaluator.evaluate_single(x, problem.sample_data)


def _build_best_meta(
    pareto_idx,
    sample_data,
    genome,
    F_vec,
    cached,
    iou_0,
    early_stopped,
    early_stop_gen,
    problem,
    runtime,
):
    """Assemble the full metadata dict for the best (minimum-L2) Pareto solution.

    :param pareto_idx: Index of this solution in the Pareto front.
    :param sample_data: Sample dict with data-source and ground-truth info.
    :param genome: Decoded genome dict from :func:`~search.utils._genome.decode_genome`.
    :param F_vec: Objective vector ``[iou, img_dist, txt_dist]``.
    :param cached: Metrics dict from the cache for this genome.
    :param iou_0: Baseline (clean-image) IoU.
    :param early_stopped: Whether the search terminated early.
    :param early_stop_gen: Generation at which early stop was triggered (or ``None``).
    :param problem: :class:`~search._problem.PerturbationProblem` instance.
    :param runtime: Wall-clock seconds for the full search.
    :returns: Serialisable metadata dict.
    """
    iou_adv, img_dist, txt_dist = float(F_vec[0]), float(F_vec[1]), float(F_vec[2])
    return {
        "data_source": {
            "folder_path": sample_data["folder_path"],
            "folder_id": sample_data["folder_id"],
            "category": sample_data["category"],
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
            "txt_sim": float(f"{1.0 - txt_dist:.5f}"),
        },
        "baseline_iou": float(f"{iou_0:.5f}"),
        "l2_distance": float(f"{np.linalg.norm(F_vec):.5f}"),
        "original_prompt": sample_data["original_prompt"],
        "ground_truth_bboxes": sample_data["gt_bboxes"],
        "vlm_output": {
            "perturbed_prompt": cached["corrupt_prompt"],
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
    result, sample_data, problem, output_dir, runtime, early_stopped=False, early_stop_gen=None
):
    """Persist the full Pareto front, best-result JSON, and best perturbed image to disk.

    :param result: pymoo optimisation result object.
    :param sample_data: Sample dict with data-source and ground-truth info.
    :param problem: :class:`~search._problem.PerturbationProblem` instance.
    :param output_dir: Directory in which to write output files.
    :param runtime: Total wall-clock seconds for this sample's search.
    :param early_stopped: Whether the search terminated early.
    :param early_stop_gen: Generation at which early stop was triggered (or ``None``).
    """
    os.makedirs(output_dir, exist_ok=True)
    evaluator = problem.evaluator

    pareto_X = result.X if result.X.ndim > 1 else result.X.reshape(1, -1)
    pareto_F = result.F if result.F.ndim > 1 else result.F.reshape(1, -1)

    iou_0 = sample_data["baseline_iou"]

    pareto_records = []
    for i in range(len(pareto_X)):
        cached = _retrieve_cached_metrics(problem, pareto_X[i])
        genome = decode_genome(
            pareto_X[i],
            problem.budget_max,
            problem.mode,
            evaluator.image_perturbations,
            evaluator.text_perturbations,
        )
        iou_adv, img_dist, txt_dist = (
            float(pareto_F[i, 0]),
            float(pareto_F[i, 1]),
            float(pareto_F[i, 2]),
        )
        pareto_records.append(
            {
                "index": i,
                "genome": genome,
                "objectives": {
                    "iou": float(f"{iou_adv:.5f}"),
                    "img_dist": float(f"{img_dist:.5f}"),
                    "txt_dist": float(f"{txt_dist:.5f}"),
                    "txt_sim": float(f"{1.0 - txt_dist:.5f}"),
                },
                "baseline_iou": float(f"{iou_0:.5f}"),
                "l2_distance": float(f"{np.linalg.norm(pareto_F[i]):.5f}"),
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
        )

    pareto_output = {
        "data_source": {
            "folder_path": sample_data["folder_path"],
            "folder_id": sample_data["folder_id"],
            "category": sample_data["category"],
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

    front_path = os.path.join(output_dir, _PARETO_FILE)
    with open(front_path, "w") as f:
        json.dump(pareto_output, f, indent=4)
    logger.info("Pareto front (%d solutions) -> %s", len(pareto_records), front_path)

    best_idx = int(np.argmin(np.linalg.norm(pareto_F, axis=1)))
    best_x, best_f = pareto_X[best_idx], pareto_F[best_idx]
    best_cached = _retrieve_cached_metrics(problem, best_x)
    best_genome = decode_genome(
        best_x,
        problem.budget_max,
        problem.mode,
        evaluator.image_perturbations,
        evaluator.text_perturbations,
    )

    logger.info(
        "BEST L2 (Pareto index=%d)%s  IoU=%.5f  ImgDist=%.5f  TxtSim=%.5f  L2=%.5f",
        best_idx,
        f"  [early-stopped at gen {early_stop_gen}]" if early_stopped else "",
        best_f[0],
        best_f[1],
        1.0 - best_f[2],
        np.linalg.norm(best_f),
    )

    meta = _build_best_meta(
        pareto_idx=best_idx,
        sample_data=sample_data,
        genome=best_genome,
        F_vec=best_f,
        cached=best_cached,
        iou_0=iou_0,
        early_stopped=early_stopped,
        early_stop_gen=early_stop_gen,
        problem=problem,
        runtime=runtime,
    )
    with open(os.path.join(output_dir, _BEST_FILE), "w") as f:
        json.dump(meta, f, indent=4)

    _recreate_perturbed(best_x, sample_data, evaluator).save(
        os.path.join(output_dir, _BEST_IMAGE_FILE)
    )
