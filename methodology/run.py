import argparse
import logging
import os
import random
import time

import numpy as np
import torch

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import search._config as _search_cfg
from vlm import Qwen3VLInstance, GemmaVLInstance, HunyuanVLInstance, KimiVLInstance
from search import (
    FitnessEvaluator,
    AdversarialProblem,
    BudgetRepair, BudgetAwareSampling, EarlyStopCallback,
    load_sample, get_all_sample_folders, get_output_dir,
    is_already_processed, save_baseline_fail, save_all_meta,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal adversarial search")

    parser.add_argument("--vlm", required=True, choices=["qwen", "gemma", "hunyuan", "kimi"],
                        help="Vision-language model to attack")
    parser.add_argument("--mode", choices=["multi", "image", "text"], default="multi",
                        help="Corruption modality (default: %(default)s)")
    parser.add_argument("--results-dir", default=_search_cfg.RESULTS_DIR,
                        help="Directory with selected samples (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="Root directory for output results (default: auto from mode)")
    parser.add_argument("--seed", type=int, default=42669,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--pop-size", type=int, default=30,
                        help="NSGA-II population size (default: %(default)s)")
    parser.add_argument("--num-generations", type=int, default=15,
                        help="Number of NSGA-II generations (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=15,
                        help="VLM evaluation batch size (default: %(default)s)")
    parser.add_argument("--budget-max", type=float, default=1.0,
                        help="Max corruption budget per modality (default: %(default)s)")
    parser.add_argument("--baseline-iou-min", type=float, default=0.5,
                        help="Min clean-image IoU threshold (default: %(default)s)")
    parser.add_argument("--early-stop-iou-max", type=float, default=0.35,
                        help="IoU threshold for early stop (default: %(default)s)")
    parser.add_argument("--early-stop-img-dist-max", type=float, default=0.1,
                        help="Image distance threshold for early stop (default: %(default)s)")
    parser.add_argument("--early-stop-txt-sim-min", type=float, default=0.70,
                        help="Text similarity threshold for early stop (default: %(default)s)")
    parser.add_argument("--max-resolution", type=int, default=1024,
                        help="Max image side length before resizing (default: %(default)s)")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    if args.output_dir is None:
        args.output_dir = _search_cfg.OUTPUT_BASE_DIRS[args.mode]

    _search_cfg.RESULTS_DIR = args.results_dir
    _search_cfg.OUTPUT_BASE_DIRS[args.mode] = args.output_dir

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _VLM_MAP = {
        "qwen": Qwen3VLInstance,
        "gemma": GemmaVLInstance,
        "hunyuan": HunyuanVLInstance,
        "kimi": KimiVLInstance,
    }
    vlm = _VLM_MAP[args.vlm](seed=args.seed)

    # Ugly patching the results dir.
    _search_cfg.OUTPUT_BASE_DIRS[args.mode] = args.output_dir = f"results/{args.vlm}/{args.output_dir}"

    all_samples = get_all_sample_folders()

    if not all_samples:
        logger.info("Nothing to process. Exiting.")
        return
    logger.info("Found %d valid sample folders.", len(all_samples))

    pending = []
    skipped = 0
    for folder_path, category, folder_id in all_samples:
        if is_already_processed(category, folder_id, mode=args.mode):
            skipped += 1
        else:
            pending.append((folder_path, category, folder_id))

    if not pending:
        logger.info("All samples already processed. Exiting.")
        return
    logger.info("Skipping %d already-processed samples. %d remaining.", skipped, len(pending))

    evaluator = FitnessEvaluator(seed=args.seed, vlm=vlm, mode=args.mode)

    first_sample_data = load_sample(pending[0][0], max_resolution=args.max_resolution)
    first_sample_data["category"] = pending[0][1]
    first_sample_data["folder_id"] = pending[0][2]
    problem = AdversarialProblem(
        evaluator, first_sample_data,
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
        logger.info("SAMPLE %d/%d  %s  (%s)", sample_idx + 1, len(pending), sample_label, folder_path)

        try:
            sample_data = load_sample(folder_path, max_resolution=args.max_resolution)
            sample_data["category"] = category
            sample_data["folder_id"] = folder_id
        except Exception as e:
            logger.warning("Failed to load %s: %s. Skipping.", sample_label, e)
            continue

        baseline_iou = evaluator.evaluate_baseline(sample_data)
        sample_data["baseline_iou"] = baseline_iou

        if baseline_iou < args.baseline_iou_min:
            baseline_fail_count += 1
            logger.info(
                "%s  baseline IoU=%.5f < %.2f — VLM fails on clean input.",
                sample_label, baseline_iou, args.baseline_iou_min,
            )
            save_baseline_fail(category, folder_id, args.mode, baseline_iou, sample_data)
            continue

        logger.info("%s  baseline IoU=%.5f", sample_label, baseline_iou)

        problem.reset(sample_data)
        early_stop_cb = EarlyStopCallback(problem)

        algorithm = NSGA2(
            pop_size=args.pop_size,
            sampling=BudgetAwareSampling(budget_max=args.budget_max, mode=args.mode),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            repair=BudgetRepair(budget_max=args.budget_max, mode=args.mode),
            eliminate_duplicates=True,
        )

        t_start = time.time()
        result = minimize(
            problem, algorithm, get_termination("n_gen", args.num_generations),
            seed=args.seed, verbose=False, callback=early_stop_cb,
        )
        t_elapsed = time.time() - t_start

        did_early_stop = problem.early_stop_triggered
        stop_gen = early_stop_cb.trigger_gen

        if did_early_stop:
            early_stop_count += 1
            logger.info(
                "%s early-stopped at gen %d  (%d evals, %d skipped)  %.1fs",
                sample_label, stop_gen, problem._eval_count, problem._skipped_count, t_elapsed,
            )
        else:
            logger.info(
                "%s completed %d gens  (%d evals)  %.1fs",
                sample_label, args.num_generations, problem._eval_count, t_elapsed,
            )

        save_all_meta(
            result, sample_data, problem,
            get_output_dir(category, folder_id, mode=args.mode),
            runtime=t_elapsed,
            early_stopped=did_early_stop, early_stop_gen=stop_gen,
        )

    searched = len(pending) - baseline_fail_count
    logger.info(
        "ALL DONE  searched: %d/%d  baseline-fail: %d/%d  early-stopped: %d/%d  results: %s/",
        searched, len(pending),
        baseline_fail_count, len(pending),
        early_stop_count, searched,
        args.output_dir,
    )

if __name__ == "__main__":
    main()