import logging
import numpy as np

from pymoo.core.problem import Problem

from . import _config as _cfg
from .utils._metrics import _is_perfect

logger = logging.getLogger(__name__)


class AdversarialProblem(Problem):
    """
    Continuous multi-objective problem for NSGA-II.

    Decision variables (N_VAR floats, all in [0, 1]):
        x[0 .. N_IMG-1]            image corruption scales
        x[N_IMG .. N_IMG+N_TXT-1]  text corruption scales

    Constraints (handled by BudgetRepair, not as pymoo constraints):
        sum(x[0..N_IMG-1])   <= budget_max
        sum(x[N_IMG..end])   <= budget_max

    Objectives (all minimised):
        F1 = IoU
        F2 = Image Distance
        F3 = 1 - Text Similarity

    Early stopping: when a perfect individual is found inside a batch,
    the rest of that batch is still recorded but all subsequent batches
    are skipped. The EarlyStopCallback then prevents the next generation.
    """

    def __init__(
        self,
        evaluator,
        sample_data,
        mode: str = "multi",
        batch_size=15,
        budget_max=1.0,
        early_stop_iou_max=0.35,
        early_stop_img_dist_max=0.1,
        early_stop_txt_sim_min=0.70,
        **kwargs,
    ):
        _n_var = {"multi": _cfg.N_VAR, "image": _cfg.N_IMG, "text": _cfg.N_TXT}[mode]
        super().__init__(
            n_var=_n_var,
            n_obj=_cfg.N_OBJ,
            n_ieq_constr=0,
            xl=np.zeros(_n_var),
            xu=np.ones(_n_var),
            **kwargs,
        )
        self.mode = mode
        self.evaluator = evaluator
        self.sample_data = sample_data
        self.batch_size = batch_size
        self.budget_max = budget_max
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
        F = np.full((n, _cfg.N_OBJ), 1.0)

        for batch_start in range(0, n, self.batch_size):
            if self.early_stop_triggered:
                remaining = n - batch_start
                self._skipped_count += remaining
                break

            batch_end = min(batch_start + self.batch_size, n)
            batch_X = [X[i] for i in range(batch_start, batch_end)]

            metrics_list = self.evaluator.evaluate_batch(
                batch_X, self.sample_data
            )

            for j, metrics in enumerate(metrics_list):
                global_idx = batch_start + j
                x = X[global_idx]

                self.metrics_cache[self._cache_key(x)] = metrics
                self._eval_count += 1

                iou = metrics["iou"]
                img_dist = metrics["img_dist"]
                txt_sim = metrics["txt_sim"]

                F[global_idx] = [iou, img_dist, 1.0 - txt_sim]

                if not self.early_stop_triggered and _is_perfect(
                    iou, img_dist, txt_sim,
                    self._iou_max, self._img_dist_max, self._txt_sim_min,
                ):
                    self.early_stop_triggered = True
                    self._early_stop_eval_id = self._eval_count
                    logger.info(
                        "EARLY STOP: perfect adversarial at eval #%d "
                        "(IoU=%.5f  ImgDist=%.5f  TxtSim=%.5f). "
                        "Remaining batches in this generation will be skipped.",
                        self._eval_count, iou, img_dist, txt_sim,
                    )
                    # NOTE: we do NOT break here, the rest of the current
                    # batch was already computed so we record their metrics.

        out["F"] = F
