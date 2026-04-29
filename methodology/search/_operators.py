import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling


class BudgetRepair(Repair):
    def __init__(self, budget_max=1.0, mode="multi", n_img=0, n_txt=0):
        super().__init__()
        self.budget_max = budget_max
        self.mode = mode
        self.n_img = n_img
        self.n_txt = n_txt

    def _do(self, problem, X, **kwargs):
        np.clip(X, 0.0, 1.0, out=X)
        if self.mode == "multi":
            for block in (X[:, : self.n_img], X[:, self.n_img :]):
                sums = block.sum(axis=1, keepdims=True)
                over = (sums > self.budget_max).flatten()
                if over.any():
                    block[over] = block[over] / sums[over] * self.budget_max
        else:
            sums = X.sum(axis=1, keepdims=True)
            over = (sums > self.budget_max).flatten()
            if over.any():
                X[over] = X[over] / sums[over] * self.budget_max
        return X


class BudgetAwareSampling(Sampling):
    def __init__(self, budget_max=1.0, mode="multi", n_img=0, n_txt=0):
        super().__init__()
        self.budget_max = budget_max
        self.mode = mode
        self.n_img = n_img
        self.n_txt = n_txt

    def _do(self, problem, n_samples, **kwargs):
        n_var = {"multi": self.n_img + self.n_txt, "image": self.n_img, "text": self.n_txt}[
            self.mode
        ]
        X = np.zeros((n_samples, n_var))
        if self.mode == "multi":
            for i in range(n_samples):
                if self.n_img > 0:
                    X[i, : self.n_img] = np.random.dirichlet(
                        np.ones(self.n_img)
                    ) * np.random.uniform(0, self.budget_max)
                if self.n_txt > 0:
                    X[i, self.n_img :] = np.random.dirichlet(
                        np.ones(self.n_txt)
                    ) * np.random.uniform(0, self.budget_max)
        else:
            n = self.n_img if self.mode == "image" else self.n_txt
            for i in range(n_samples):
                X[i] = np.random.dirichlet(np.ones(n)) * np.random.uniform(0, self.budget_max)
        return X


class EarlyStopCallback(Callback):
    def __init__(self, problem_ref):
        super().__init__()
        self.problem_ref = problem_ref
        self.trigger_gen = None

    def notify(self, algorithm):
        if self.trigger_gen is not None:
            return
        if self.problem_ref.early_stop_triggered:
            self.trigger_gen = algorithm.n_gen
            algorithm.termination.force_termination = True
