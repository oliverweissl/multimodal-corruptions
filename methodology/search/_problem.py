import logging

import numpy as np
from config.experiment import (
    BATCH_SIZE,
    BUDGET_MAX,
    EARLY_STOP_IMG_DIST_MAX,
    EARLY_STOP_IOU_MAX,
    EARLY_STOP_TXT_SIM_MIN,
    N_OBJ,
)
from pymoo.core.problem import Problem

from .utils import is_perfect

logger = logging.getLogger(__name__)


class PerturbationProblem(Problem):
    """
    Continuous multi-objective problem for NSGA-II.

    Decision variables (N_VAR floats in [0, 1]):
        x[0 .. n_img-1]           image perturbation scales
        x[n_img .. n_img+n_txt-1] text perturbation scales

    Objectives (all minimised): IoU, Image Distance, 1 - Text Similarity

    Early stopping: when a perfect individual is found inside a batch,
    the rest of that batch is still recorded but subsequent batches are skipped.
    """

    def __init__(
        self,
        evaluator,
        sample_data,
        mode="multi",
        batch_size=BATCH_SIZE,
        budget_max=BUDGET_MAX,
        early_stop_iou_max=EARLY_STOP_IOU_MAX,
        early_stop_img_dist_max=EARLY_STOP_IMG_DIST_MAX,
        early_stop_txt_sim_min=EARLY_STOP_TXT_SIM_MIN,
        **kwargs,
    ):
        n_img = evaluator.n_img
        n_txt = evaluator.n_txt
        n_var = {"multi": n_img + n_txt, "image": n_img, "text": n_txt}[mode]
        super().__init__(
            n_var=n_var,
            n_obj=N_OBJ,
            n_ieq_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
            **kwargs,
        )
        self.mode = mode
        self.evaluator = evaluator
        self.sample_data = sample_data
        self.batch_size = batch_size
        self.budget_max = budget_max
        self.n_img = n_img
        self.n_txt = n_txt
        self._iou_max = early_stop_iou_max
        self._img_dist_max = early_stop_img_dist_max
        self._txt_sim_min = early_stop_txt_sim_min
        self._eval_count = 0
        self._skipped_count = 0
        self.metrics_cache = {}
        self.early_stop_triggered = False
        self._early_stop_eval_id = None

    @staticmethod
    def _cache_key(x):
        return tuple(round(float(v), 6) for v in x)

    def reset(self, sample_data):
        self.sample_data = sample_data
        self._eval_count = 0
        self._skipped_count = 0
        self.metrics_cache.clear()
        self.early_stop_triggered = False
        self._early_stop_eval_id = None

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.full((n, N_OBJ), 1.0)
        effective_batch = n if self.batch_size == 0 else self.batch_size

        for batch_start in range(0, n, effective_batch):
            if self.early_stop_triggered:
                self._skipped_count += n - batch_start
                break

            batch_end = min(batch_start + effective_batch, n)
            metrics_list = self.evaluator.evaluate_batch(
                [X[i] for i in range(batch_start, batch_end)], self.sample_data
            )

            for j, metrics in enumerate(metrics_list):
                global_idx = batch_start + j
                self.metrics_cache[self._cache_key(X[global_idx])] = metrics
                self._eval_count += 1

                iou, img_dist, txt_sim = metrics["iou"], metrics["img_dist"], metrics["txt_sim"]
                F[global_idx] = [iou, img_dist, 1.0 - txt_sim]

                if not self.early_stop_triggered and is_perfect(
                    iou,
                    img_dist,
                    txt_sim,
                    self._iou_max,
                    self._img_dist_max,
                    self._txt_sim_min,
                ):
                    self.early_stop_triggered = True
                    self._early_stop_eval_id = self._eval_count
                    logger.info(
                        "EARLY STOP: perfect test case at eval #%d "
                        "(IoU=%.5f  ImgDist=%.5f  TxtSim=%.5f). "
                        "Remaining batches in this generation will be skipped.",
                        self._eval_count,
                        iou,
                        img_dist,
                        txt_sim,
                    )

        out["F"] = F
