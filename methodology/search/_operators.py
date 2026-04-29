import numpy as np

from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling

from . import _config as _cfg


class BudgetRepair(Repair):
    """
    Clips all genes to [0, 1] and proportionally rescales each modality's
    scales if their sum exceeds budget_max.

    For multi mode, enforces budget separately on image and text blocks.
    For image/text mode, enforces budget on the single block.
    """

    def __init__(self, budget_max=1.0, mode: str = "multi"):
        super().__init__()
        self.budget_max = budget_max
        self.mode = mode

    def _do(self, problem, X, **kwargs):
        np.clip(X, 0.0, 1.0, out=X)

        if self.mode == "multi":
            img_block = X[:, :_cfg.N_IMG]
            img_sums = img_block.sum(axis=1, keepdims=True)
            over = (img_sums > self.budget_max).flatten()
            if over.any():
                img_block[over] = img_block[over] / img_sums[over] * self.budget_max

            txt_block = X[:, _cfg.N_IMG:]
            txt_sums = txt_block.sum(axis=1, keepdims=True)
            over = (txt_sums > self.budget_max).flatten()
            if over.any():
                txt_block[over] = txt_block[over] / txt_sums[over] * self.budget_max
        else:
            # Single-modality: enforce budget on all columns
            sums = X.sum(axis=1, keepdims=True)
            over = (sums > self.budget_max).flatten()
            if over.any():
                X[over] = X[over] / sums[over] * self.budget_max

        return X


class BudgetAwareSampling(Sampling):
    """
    Initial population sampler that produces diverse budget-usage levels.

    Picks a budget fraction t ~ Uniform(0, budget_max) per individual per
    modality, then distributes via Dirichlet(1,...,1). This gives coverage
    from near-zero to full budget, unlike FloatRandomSampling which would
    always saturate the budget after BudgetRepair.

    For multi mode, samples image and text blocks independently.
    For image/text mode, samples a single block of the appropriate size.
    """

    def __init__(self, budget_max=1.0, mode: str = "multi"):
        super().__init__()
        self.budget_max = budget_max
        self.mode = mode

    def _do(self, problem, n_samples, **kwargs):
        _n_var = {"multi": _cfg.N_VAR, "image": _cfg.N_IMG, "text": _cfg.N_TXT}[self.mode]
        X = np.zeros((n_samples, _n_var))

        if self.mode == "multi":
            for i in range(n_samples):
                t_img = np.random.uniform(0.0, self.budget_max)
                if _cfg.N_IMG > 0:
                    fracs_img = np.random.dirichlet(np.ones(_cfg.N_IMG))
                    X[i, :_cfg.N_IMG] = fracs_img * t_img

                t_txt = np.random.uniform(0.0, self.budget_max)
                if _cfg.N_TXT > 0:
                    fracs_txt = np.random.dirichlet(np.ones(_cfg.N_TXT))
                    X[i, _cfg.N_IMG:] = fracs_txt * t_txt
        else:
            n_attacks = _cfg.N_IMG if self.mode == "image" else _cfg.N_TXT
            for i in range(n_samples):
                t = np.random.uniform(0.0, self.budget_max)
                fracs = np.random.dirichlet(np.ones(n_attacks))
                X[i] = fracs * t

        return X


class EarlyStopCallback(Callback):
    def __init__(self, problem_ref):
        super().__init__()
        self.problem_ref = problem_ref
        self.trigger_gen = None

    @property
    def found_perfect(self):
        return self.problem_ref.early_stop_triggered

    def notify(self, algorithm):
        if self.trigger_gen is not None:
            return
        if self.problem_ref.early_stop_triggered:
            self.trigger_gen = algorithm.n_gen
            algorithm.termination.force_termination = True
