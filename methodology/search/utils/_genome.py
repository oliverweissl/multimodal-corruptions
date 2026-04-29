from .. import _config as _cfg


def decode_genome(x, budget_max, mode="multi"):
    """Convert a raw decision vector into a human-readable dict."""
    if mode == "image":
        img_scales = x
        txt_scales = []
    elif mode == "text":
        img_scales = []
        txt_scales = x
    else:  # multi
        img_scales = x[:_cfg.N_IMG]
        txt_scales = x[_cfg.N_IMG:]

    return {
        "image_corruptions": {
            name: float(f"{img_scales[i]:.6f}")
            for i, name in enumerate(_cfg.IMAGE_ATTACKS)
            if i < len(img_scales)
        },
        "text_corruptions": {
            name: float(f"{txt_scales[i]:.6f}")
            for i, name in enumerate(_cfg.TEXT_ATTACKS)
            if i < len(txt_scales)
        },
        "img_budget_used": float(f"{sum(img_scales):.6f}"),
        "txt_budget_used": float(f"{sum(txt_scales):.6f}"),
        "budget_max": budget_max,
        "mode": mode,
    }


def _active_corruptions_str(x, mode="multi"):
    """One-line summary of corruptions with scale > 0."""
    if mode == "image":
        img_scales = x
        txt_scales = []
    elif mode == "text":
        img_scales = []
        txt_scales = x
    else:  # multi
        img_scales = x[:_cfg.N_IMG]
        txt_scales = x[_cfg.N_IMG:]

    parts = []
    for i, name in enumerate(_cfg.IMAGE_ATTACKS):
        if i < len(img_scales) and img_scales[i] > 0:
            parts.append(f"{name}={img_scales[i]:.3f}")
    img_str = "+".join(parts) if parts else "none"

    parts = []
    for i, name in enumerate(_cfg.TEXT_ATTACKS):
        if i < len(txt_scales) and txt_scales[i] > 0:
            parts.append(f"{name}={txt_scales[i]:.3f}")
    txt_str = "+".join(parts) if parts else "none"

    return f"Img[{img_str}] Txt[{txt_str}]"
