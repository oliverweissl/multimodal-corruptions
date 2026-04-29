import numpy as np


def decode_genome(
    x: np.ndarray,
    budget_max: float,
    mode: str,
    image_perturbations: list[str],
    text_perturbations: list[str],
) -> dict:
    """Convert a raw decision vector into a human-readable corruption dict.

    :param x: Decision vector of floats in [0, 1].
    :param budget_max: Maximum allowed budget per modality.
    :param mode: Search mode — ``'multi'``, ``'image'``, or ``'text'``.
    :param image_perturbations: Ordered list of image perturbation names.
    :param text_perturbations: Ordered list of text perturbation names.
    :returns: Dict with image_corruptions, text_corruptions, budget usage, and mode.
    """
    n_img = len(image_perturbations)
    if mode == "image":
        img_scales, txt_scales = x, []
    elif mode == "text":
        img_scales, txt_scales = [], x
    else:
        img_scales, txt_scales = x[:n_img], x[n_img:]

    return {
        "image_corruptions": {
            name: float(f"{img_scales[i]:.6f}")
            for i, name in enumerate(image_perturbations)
            if i < len(img_scales)
        },
        "text_corruptions": {
            name: float(f"{txt_scales[i]:.6f}")
            for i, name in enumerate(text_perturbations)
            if i < len(txt_scales)
        },
        "img_budget_used": float(f"{sum(img_scales):.6f}"),
        "txt_budget_used": float(f"{sum(txt_scales):.6f}"),
        "budget_max": budget_max,
        "mode": mode,
    }
