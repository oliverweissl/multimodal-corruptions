"""Shared data loading utilities for the RQ analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / "results"
MODELS = ["deepseek", "gemma", "intern", "kimi", "qwen"]

IMG_CORRUPTIONS = [
    "elastic", "gaussian_noise", "defocus_blur", "motion_blur", "fog_filter",
    "snow_filter", "contrast", "false_color", "grayscale", "cutout",
    "pixelate", "jpeg_filter",
]
TXT_CORRUPTIONS = [
    "homophone", "synonym", "ata_saliency", "fragmentation", "character_noise",
]


def _parse_best_result(path: Path, model: str) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    # resolve modality (multimodal / unimodal) and sub-mode from path
    parts = path.parts
    # …/results/<model>/<modality>/<subdir…>/<id>/best_result.json
    model_idx = next((i for i, p in enumerate(parts) if p == model), None)
    if model_idx is None:
        return None

    modality = parts[model_idx + 1] if model_idx + 1 < len(parts) else "unknown"

    # genome mode (multi / image / text)
    genome = d.get("genome", {})
    genome_mode = genome.get("mode", "unknown")

    # determine object-count category from folder path
    data_src = d.get("data_source", {})
    folder_path = data_src.get("folder_path", "")
    if "single" in folder_path:
        if "solo" in folder_path:
            obj_category = "single/solo"
        else:
            obj_category = "single/multi"
    else:
        obj_category = "multi"

    # objectives
    objs = d.get("objectives", {})
    baseline_iou = d.get("baseline_iou", float("nan"))
    final_iou = objs.get("iou", float("nan"))

    # applied corruptions (use genome values if applied_ keys absent)
    applied_img = d.get("applied_img_corruptions", genome.get("image_corruptions", {}))
    applied_txt = d.get("applied_txt_corruptions", genome.get("text_corruptions", {}))

    row: dict[str, Any] = {
        "model": model,
        "modality": modality,
        "genome_mode": genome_mode,       # multi / image / text
        "obj_category": obj_category,     # multi / single/solo / single/multi
        "folder_id": data_src.get("folder_id", path.parent.name),
        "filename": data_src.get("filename", ""),
        "status": "success",
        "baseline_iou": baseline_iou,
        "final_iou": final_iou,
        "iou_reduction": baseline_iou - final_iou,
        "img_dist": objs.get("img_dist", float("nan")),
        "txt_dist": objs.get("txt_dist", float("nan")),
        "txt_sim": objs.get("txt_sim", float("nan")),
        "img_budget_used": genome.get("img_budget_used", float("nan")),
        "txt_budget_used": genome.get("txt_budget_used", float("nan")),
        "budget_max": genome.get("budget_max", 1.0),
        "runtime": d.get("runtime", float("nan")),
        "total_evaluations": d.get("total_evaluations", float("nan")),
        "skipped_evaluations": d.get("skipped_evaluations", 0),
        "early_stopped": d.get("early_stopped", False),
        "early_stop_generation": d.get("early_stop_generation"),
        "pareto_index": d.get("pareto_index", 0),
        # image path for pixel-level metrics
        "_best_img_path": str(path.parent / "best_result.png"),
        "_orig_img_folder": folder_path,
    }

    # per-corruption intensities
    for c in IMG_CORRUPTIONS:
        row[f"img_{c}"] = applied_img.get(c, 0.0)
    for c in TXT_CORRUPTIONS:
        row[f"txt_{c}"] = applied_txt.get(c, 0.0)

    # perturbed prompt for text metrics
    vlm = d.get("vlm_output", {})
    row["original_prompt"] = d.get("original_prompt", "")
    row["perturbed_prompt"] = vlm.get("perturbed_prompt", "")

    return row


def _parse_baseline_fail(path: Path, model: str) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    parts = path.parts
    model_idx = next((i for i, p in enumerate(parts) if p == model), None)
    if model_idx is None:
        return None
    modality = parts[model_idx + 1] if model_idx + 1 < len(parts) else "unknown"

    data_src = d.get("data_source", {})
    folder_path = data_src.get("folder_path", "")
    if "single" in folder_path:
        if "solo" in folder_path:
            obj_category = "single/solo"
        else:
            obj_category = "single/multi"
    else:
        obj_category = "multi"

    # Infer genome_mode from the result directory structure
    # …/results/<model>/<modality>/image/… → image; /multi/… → multi; /text/… → text
    if "unimodal" in str(path):
        # sub-folder after unimodal
        uni_idx = next((i for i, p in enumerate(parts) if p == "unimodal"), None)
        genome_mode = parts[uni_idx + 1] if uni_idx and uni_idx + 1 < len(parts) else "image"
    else:
        genome_mode = "multi"

    baseline_iou = d.get("baseline_iou", float("nan"))

    return {
        "model": model,
        "modality": modality,
        "genome_mode": genome_mode,
        "obj_category": obj_category,
        "folder_id": data_src.get("folder_id", path.parent.name),
        "filename": data_src.get("filename", ""),
        "status": "baseline_fail",
        "baseline_iou": baseline_iou,
        "final_iou": baseline_iou,
        "iou_reduction": 0.0,
        "img_dist": 0.0,
        "txt_dist": 0.0,
        "txt_sim": 1.0,
        "img_budget_used": float("nan"),
        "txt_budget_used": float("nan"),
        "budget_max": float("nan"),
        "runtime": float("nan"),
        "total_evaluations": float("nan"),
        "skipped_evaluations": float("nan"),
        "early_stopped": False,
        "early_stop_generation": None,
        "pareto_index": float("nan"),
        "_best_img_path": "",
        "_orig_img_folder": folder_path,
        "original_prompt": d.get("original_prompt", ""),
        "perturbed_prompt": "",
        **{f"img_{c}": 0.0 for c in IMG_CORRUPTIONS},
        **{f"txt_{c}": 0.0 for c in TXT_CORRUPTIONS},
    }


def load_all_results(
    models: list[str] | None = None,
    include_baseline_fail: bool = True,
) -> pd.DataFrame:
    """Load all best_result.json and optionally baseline_fail.json files.

    Returns a DataFrame with one row per test case.
    Columns include model, modality, genome_mode, obj_category, IoU metrics,
    distortion metrics, budget, runtime, per-corruption intensities.
    """
    if models is None:
        models = MODELS

    rows: list[dict] = []
    for model in models:
        model_dir = RESULTS_ROOT / model
        if not model_dir.exists():
            continue
        for p in model_dir.rglob("best_result.json"):
            row = _parse_best_result(p, model)
            if row:
                rows.append(row)
        if include_baseline_fail:
            for p in model_dir.rglob("baseline_fail.json"):
                row = _parse_baseline_fail(p, model)
                if row:
                    rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # convenience label columns
    df["modality_label"] = df["modality"].map(
        {"multimodal": "Multimodal", "unimodal": "Unimodal"}
    ).fillna(df["modality"])

    df["mode_label"] = df["genome_mode"].map(
        {"multi": "Multi (img+txt)", "image": "Image only", "text": "Text only"}
    ).fillna(df["genome_mode"])

    df["category_label"] = df["obj_category"].map(
        {"multi": "Multi-object", "single/solo": "Single-object (solo)",
         "single/multi": "Single-object (multi)"}
    ).fillna(df["obj_category"])

    # compound split key for easy grouping
    df["split"] = df["modality"] + "/" + df["genome_mode"] + "/" + df["obj_category"]

    return df


def success_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where optimization actually ran (no baseline_fail)."""
    return df[df["status"] == "success"].copy()
