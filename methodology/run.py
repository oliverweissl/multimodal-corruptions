import argparse
import logging
import os
import random
import time

import config.experiment as _exp
import config.paths as _paths
import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from search import (
    BudgetAwareSampling,
    BudgetRepair,
    EarlyStopCallback,
    FitnessEvaluator,
    PerturbationProblem,
    get_all_sample_folders,
    get_output_dir,
    is_already_processed,
    load_sample,
    save_all_meta,
    save_baseline_fail,
)
from vlm import GemmaVLInstance, KimiVLInstance, Qwen3VLInstance, DeepSeekVL2Instance, InternVL3Instance
_VLM_MAP = {
        "qwen": Qwen3VLInstance,  # Works on vLLM
        "gemma": GemmaVLInstance,  # Works on vLLM
        "kimi": KimiVLInstance,  # Works on vLLM
        #"hunyuan": HunyuanVLInstance,  # Not working on vLLM
        "deepseek": DeepSeekVL2Instance,  # Works on vLLM
        "intern": InternVL3Instance, # Not working on vLLM
    }

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the perturbation robustness search experiment.

    :returns: Populated :class:`argparse.Namespace` with all experiment parameters.
    """
    parser = argparse.ArgumentParser(description="Multimodal perturbation robustness evaluation")

    parser.add_argument(
        "--vlm",
        required=True,
        choices=_VLM_MAP.keys(),
        help="Vision-language model to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=["multi", "image", "text"],
        default="multi",
        help="Perturbation modality (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        default=_paths.RESULTS_DIR,
        help="Directory with selected samples (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for output results (default: auto from mode)",
    )
    parser.add_argument(
        "--seed", type=int, default=_exp.SEED, help="Random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=_exp.POP_SIZE,
        help="NSGA-II population size (default: %(default)s)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=_exp.NUM_GENERATIONS,
        help="Number of NSGA-II generations (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_exp.BATCH_SIZE,
        help="VLM evaluation batch size; 0 = whole population at once (default: %(default)s)",
    )
    parser.add_argument(
        "--budget-max",
        type=float,
        default=_exp.BUDGET_MAX,
        help="Max perturbation budget per modality (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-iou-min",
        type=float,
        default=_exp.BASELINE_IOU_MIN,
        help="Min clean-image IoU threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--early-stop-iou-max",
        type=float,
        default=_exp.EARLY_STOP_IOU_MAX,
        help="IoU threshold for early stop (default: %(default)s)",
    )
    parser.add_argument(
        "--early-stop-img-dist-max",
        type=float,
        default=_exp.EARLY_STOP_IMG_DIST_MAX,
        help="Image distance threshold for early stop (default: %(default)s)",
    )
    parser.add_argument(
        "--early-stop-txt-sim-min",
        type=float,
        default=_exp.EARLY_STOP_TXT_SIM_MIN,
        help="Text similarity threshold for early stop (default: %(default)s)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=_exp.MAX_RESOLUTION,
        help="Max image side length before resizing (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: run NSGA-II-based multimodal perturbation search over all pending samples."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    output_base = args.output_dir or _paths.OUTPUT_BASE_DIRS[args.mode]
    output_dir = os.path.join("results", args.vlm, output_base)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vlm = _VLM_MAP[args.vlm](seed=args.seed)

    all_samples = get_all_sample_folders(args.results_dir)
    if not all_samples:
        logger.info("Nothing to process. Exiting.")
        return
    logger.info("Found %d valid sample folders.", len(all_samples))

    pending = []
    skipped = 0
    for folder_path, category, folder_id in all_samples:
        if is_already_processed(category, folder_id, output_dir):
            skipped += 1
        else:
            pending.append((folder_path, category, folder_id))

    if not pending:
        logger.info("All samples already processed. Exiting.")
        return
    logger.info("Skipping %d already-processed samples. %d remaining.", skipped, len(pending))

    evaluator = FitnessEvaluator(
        vlm=vlm,
        seed=args.seed,
        mode=args.mode,
    )
    n_img, n_txt = evaluator.n_img, evaluator.n_txt

    first_sample_data = load_sample(pending[0][0], max_resolution=args.max_resolution)
    first_sample_data["category"] = pending[0][1]
    first_sample_data["folder_id"] = pending[0][2]
    problem = PerturbationProblem(
        evaluator,
        first_sample_data,
        mode=args.mode,
        batch_size=args.batch_size,
        budget_max=args.budget_max,
        early_stop_iou_max=args.early_stop_iou_max,
        early_stop_img_dist_max=args.early_stop_img_dist_max,
        early_stop_txt_sim_min=args.early_stop_txt_sim_min,
    )

    early_stop_count = 0
    baseline_fail_count = 0

    for sample_idx, (folder_path, category, folder_id) in enumerate(pending):
        sample_label = f"{category}/{folder_id}"
        logger.info(
            "SAMPLE %d/%d  %s  (%s)", sample_idx + 1, len(pending), sample_label, folder_path
        )

        try:
            sample_data = load_sample(folder_path, max_resolution=args.max_resolution)
            sample_data["category"] = category
            sample_data["folder_id"] = folder_id
        except Exception as e:
            logger.warning("Failed to load %s: %s. Skipping.", sample_label, e)
            continue

        baseline_iou = evaluator.evaluate_baseline(sample_data)
        sample_data["baseline_iou"] = baseline_iou

        sample_output_dir = get_output_dir(category, folder_id, output_dir)

        if baseline_iou < args.baseline_iou_min:
            baseline_fail_count += 1
            logger.info(
                "%s  baseline IoU=%.5f < %.2f — VLM fails on clean input.",
                sample_label,
                baseline_iou,
                args.baseline_iou_min,
            )
            save_baseline_fail(sample_output_dir, baseline_iou, sample_data)
            continue

        logger.info("%s  baseline IoU=%.5f", sample_label, baseline_iou)

        problem.reset(sample_data)
        early_stop_cb = EarlyStopCallback(problem)

        algorithm = NSGA2(
            pop_size=args.pop_size,
            sampling=BudgetAwareSampling(
                budget_max=args.budget_max, mode=args.mode, n_img=n_img, n_txt=n_txt
            ),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            repair=BudgetRepair(
                budget_max=args.budget_max, mode=args.mode, n_img=n_img, n_txt=n_txt
            ),
            eliminate_duplicates=True,
        )

        t_start = time.time()
        result = minimize(
            problem,
            algorithm,
            get_termination("n_gen", args.num_generations),
            seed=args.seed,
            verbose=False,
            callback=early_stop_cb,
        )
        t_elapsed = time.time() - t_start

        did_early_stop = problem.early_stop_triggered
        stop_gen = early_stop_cb.trigger_gen

        if did_early_stop:
            early_stop_count += 1
            logger.info(
                "%s early-stopped at gen %d  (%d evals, %d skipped)  %.1fs",
                sample_label,
                stop_gen,
                problem._eval_count,
                problem._skipped_count,
                t_elapsed,
            )
        else:
            logger.info(
                "%s completed %d gens  (%d evals)  %.1fs",
                sample_label,
                args.num_generations,
                problem._eval_count,
                t_elapsed,
            )

        save_all_meta(
            result,
            sample_data,
            problem,
            sample_output_dir,
            runtime=t_elapsed,
            early_stopped=did_early_stop,
            early_stop_gen=stop_gen,
        )

    searched = len(pending) - baseline_fail_count
    logger.info(
        "ALL DONE  searched: %d/%d  baseline-fail: %d/%d  early-stopped: %d/%d  results: %s/",
        searched,
        len(pending),
        baseline_fail_count,
        len(pending),
        early_stop_count,
        searched,
        output_dir,
    )


if __name__ == "__main__":
    main()
