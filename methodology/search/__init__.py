from ._data import (
    get_all_sample_folders,
    get_output_dir,
    is_already_processed,
    load_sample,
    save_all_meta,
    save_baseline_fail,
)
from ._operators import BudgetAwareSampling, BudgetRepair, EarlyStopCallback
from ._problem import PerturbationProblem

__all__ = [
    "FitnessEvaluator",
    "PerturbationProblem",
    "BudgetRepair",
    "BudgetAwareSampling",
    "EarlyStopCallback",
    "load_sample",
    "get_all_sample_folders",
    "get_output_dir",
    "is_already_processed",
    "save_baseline_fail",
    "save_all_meta",
]


def __getattr__(name: str):
    # Defer FitnessEvaluator import so vLLM is not loaded at package import time.
    if name == "FitnessEvaluator":
        from ._evaluator import FitnessEvaluator
        return FitnessEvaluator
    raise AttributeError(f"module 'search' has no attribute {name!r}")
